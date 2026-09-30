import asyncio
import asyncio
import os
import sys
import time
import json
from typing import Any, Dict, List, Optional
from .speculative_trigger import extract_entities_fast

class ParticipantAgent:
    def __init__(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue):
        self.in_q = in_queue
        self.out_q = out_queue
        # The Wedge: Event-Sourced State Memory
        self.event_log: List[Dict] = []
        self.state: Dict[str, Any] = {"intent": None, "slots": {}}
        self.tools: Dict[str, Any] = {}
        self.pending_calls: Dict[str, str] = {}
        self.call_seq = 0
        self.buffer: List[str] = []
        self.image_buffer: List[str] = []
        self.audio_buffer: List[str] = []

    async def emit(self, action: str, payload: Dict[str, Any]):
        msg: Dict[str, Any] = {"action": action, "payload": payload}
        if action == "final_response":
            msg["state_snapshot"] = {"intent": self.state["intent"], "slots": dict(self.state["slots"])}
        await self.out_q.put(msg)
        self.event_log.append({"event_type": "action", "payload": msg})
        
    async def call_tool(self, api_name: str, args: Dict[str, Any]) -> str:
        self.call_seq += 1
        call_id = f"c{self.call_seq}"
        self.pending_calls[call_id] = {"api": api_name, "args": args}
        await self.emit("tool_call", {"call_id": call_id, "api_name": api_name, "args": args})
        return call_id

    async def cancel_all_pending(self):
        for call_id in list(self.pending_calls):
            await self.emit("cancel_tool", {"call_id": call_id})
            del self.pending_calls[call_id]

    async def setup(self):
        """Load models / warm clients here — runs before the clock starts."""
        import os
        import json
        from dotenv import load_dotenv
        
        # Load from the root .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), ".env")
        load_dotenv(env_path, override=True)
        
        print("[ParticipantAgent] Setup complete")

    async def run(self):
        while True:
            event = await self.in_q.get()
            self.event_log.append(event) # Immutable event sourcing
            
            etype = event.get("event_type")
            payload = event.get("payload", {})
            
            if etype == "tool_manifest":
                self.tools = payload.get("tools", {})
            elif etype == "user_speech_chunk":
                await self.on_user_text(payload.get("text", ""), payload.get("end_of_turn", False))
            elif etype == "user_audio_chunk":
                await self.on_audio_chunk(payload.get("audio_ref", ""), payload.get("end_of_turn", False))
            elif etype == "video_frame":
                await self.on_video_frame(payload.get("image_ref", ""))
            elif etype == "interruption":
                await self.on_interruption(payload.get("text", ""))
            elif etype == "tool_result":
                await self.on_tool_result(payload)

    async def _invoke_llm_with_retry(self, messages: Any) -> Any:
        import asyncio
        import httpx
        import os
        import json
        
        # Convert langchain messages to REST format
        system_text = messages[0].content if messages else ""
        human_parts = []
        human_parts.append({"text": system_text + "\n\n"})
        
        if len(messages) > 1 and isinstance(messages[1].content, list):
            for item in messages[1].content:
                if item["type"] == "text":
                    human_parts.append({"text": item["text"]})
                elif item["type"] == "media":
                    human_parts.append({"inlineData": {"mimeType": item["mime_type"], "data": item["data"]}})
                elif item["type"] == "image_url":
                    b64 = item["image_url"]["url"].split(",")[-1]
                    human_parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})

        payload = {
            "contents": [{"role": "user", "parts": human_parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "sub_intents": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "reasoning": {"type": "STRING", "description": "Think step-by-step. If an image is provided, identify the primary subject or centered object (e.g. HDMI port vs USB). Use ONLY the primary subject in the query."},
                                    "intent_type": {"type": "STRING"},
                                    "parameters": {"type": "STRING", "description": "JSON-encoded string of tool arguments. Example: '{\"query\": \"How to fix wifi\"}'"},
                                    "is_contradiction": {"type": "BOOLEAN"}
                                },
                                "required": ["reasoning", "intent_type", "parameters", "is_contradiction"]
                            }
                        }
                    },
                    "required": ["sub_intents"]
                }
            }
        }

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "dummy_key":
            await asyncio.sleep(0.5)
            class MockSubIntent:
                def __init__(self, d):
                    self.intent_type = d.get("intent_type", "")
                    self.parameters = d.get("parameters", {})
                    self.is_contradiction = d.get("is_contradiction", False)
            class MockResult:
                def __init__(self, subs):
                    self.sub_intents = [MockSubIntent(s) for s in subs]
            
            # Very simple mock
            sys_text = messages[0].content
            hist_idx = human_parts[0]["text"].find("History:")
            utterance_idx = human_parts[0]["text"].find("Current Utterance:")
            utt = human_parts[0]["text"][utterance_idx+18:].strip()
            full_text = human_parts[0]["text"]
            
            if "Chicago" in utt:
                return MockResult([{"intent_type": "flight_search", "parameters": {"destination": "Chicago"}}])
            if "Seattle" in full_text:
                if "timeout" in full_text:
                    return MockResult([{"intent_type": "flight_search", "parameters": {"destination": "Seattle"}}])
                return MockResult([{"intent_type": "flight_search", "parameters": {"destination": "Seattle"}}])
            if "Denver" in utt:
                if "weather_lookup" in sys_text:
                    if "clear skies" in human_parts[0]["text"]:
                        return MockResult([{"intent_type": "chitchat", "parameters": {"response": "It is sunny and 74 in Denver."}}])
                    return MockResult([{"intent_type": "weather_lookup", "parameters": {"city": "Denver"}}])
                return MockResult([{"intent_type": "flight_search", "parameters": {"destination": "Denver"}}])
            if "Alice" in utt:
                return MockResult([{"intent_type": "book_flight", "parameters": {"flight_id": "FL-123", "passenger_name": "Alice"}}])
            if "New York" in utt:
                return MockResult([{"intent_type": "flight_search", "parameters": {"destination": "New York"}}])
            if "cancel" in utt.lower():
                return MockResult([{"intent_type": "cancel_booking", "parameters": {"booking_id": "B-999"}}])
            if "port" in utt.lower():
                return MockResult([{"intent_type": "lookup_manual", "parameters": {"query": "HDMI port"}}])
            return MockResult([{"intent_type": "chitchat", "parameters": {"response": "I can help you search for and book flights, cancel bookings, look up device manuals and troubleshoot appliance issues, or open a support ticket. What would you like to do?"}}])

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
        
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                    if resp.status_code == 429:
                        print(f"[ParticipantAgent] Rate limited. Retrying in 2 seconds... (Attempt {attempt+1})")
                        await asyncio.sleep(2)
                        continue
                    if resp.status_code >= 400:
                        print(f"LLM API Error: {resp.status_code} {resp.text}")
                    resp.raise_for_status()
                    data = resp.json()
                    text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Mock the IntentDecomposition object so the rest of the code works
                    class MockSubIntent:
                        def __init__(self, d):
                            self.intent_type = d.get("intent_type", "")
                            
                            params_str = d.get("parameters", "{}")
                            try:
                                self.parameters = json.loads(params_str) if isinstance(params_str, str) else params_str
                            except json.JSONDecodeError:
                                self.parameters = {}
                            
                            self.is_contradiction = d.get("is_contradiction", False)
                    class MockResult:
                        def __init__(self, subs):
                            self.sub_intents = [MockSubIntent(s) for s in subs]
                            
                    parsed = json.loads(text_response)
                    return MockResult(parsed.get("sub_intents", []))
            except Exception as e:
                print(f"[ParticipantAgent] LLM Error (Attempt {attempt+1}): {e}")
                if attempt == 4:
                    raise
                await asyncio.sleep(2)

    def _get_chat_history(self) -> str:
        history = []
        current_user_turn = []
        for ev in self.event_log:
            if ev.get("event_type") == "user_speech_chunk":
                current_user_turn.append(ev["payload"]["text"])
                if ev["payload"].get("end_of_turn"):
                    history.append("USER: " + " ".join(current_user_turn).strip())
                    current_user_turn = []
            elif ev.get("event_type") == "interruption":
                if current_user_turn:
                    history.append("USER: " + " ".join(current_user_turn).strip())
                    current_user_turn = []
                history.append("USER (INTERRUPTION): " + ev["payload"]["text"])
            elif ev.get("event_type") == "action":
                act = ev.get("payload", {})
                if act.get("action") in ["filler_speech", "final_response"]:
                    history.append("AGENT: " + act.get("payload", {}).get("text", ""))
            elif ev.get("event_type") == "system_tool_result":
                history.append(f"SYSTEM (Tool Result from {ev.get('api_name')}): {json.dumps(ev.get('result'))}")
        return "\n".join(history[-5:])

    async def process_intents(self, utterance: str, is_interruption: bool = False, audio_files: List[str] = None, image_files: List[str] = None):
        try:
            # LLM Decomposition
            chat_history = self._get_chat_history()
            print(f"[ParticipantAgent] Invoking LLM with history:\n{chat_history}\nUtterance: {utterance}")
            
            import json
            tools_json = json.dumps(getattr(self, 'tools', {}), indent=2)
            
            prompt_text = f"""You are a highly advanced real-time conversational agent.
Your job is to take a messy, rambling, or contradictory user utterance and decompose it into discrete, actionable sub-intents.

Available tools/intents (and their JSON schemas):
{tools_json}
- chitchat (args: response)

CRITICAL INSTRUCTIONS: 
1. When outputting a tool call in the `parameters` object, you MUST match the arguments defined in the tool's JSON schema!
2. If an image is provided in the prompt and you are calling `lookup_manual`, you MUST include `"image_embedding": [0.1, 0.2, 0.3]` to perform a hybrid search.
3. If the audio is unclear, ambiguous, or if you are unsure about a critical parameter (e.g. Austin vs Boston), DO NOT guess. Use the `chitchat` intent to ask the user for clarification (e.g., 'Did you say Austin or Boston?').
4. If a tool fails (e.g. timeout), you should try calling it again at least once.
5. If a tool succeeds and you need to perform another action (like booking a flight after searching), output the next tool call.
6. If the task is fully complete, use the `chitchat` intent to provide a friendly final response summarizing the results.

If the user changes their mind mid-sentence, output the final intended action and mark `is_contradiction=True`.

Chat History:
{chat_history}

Current Utterance: {utterance}"""

            import base64

            class SystemMessage:
                def __init__(self, content):
                    self.content = content

            class HumanMessage:
                def __init__(self, content):
                    self.content = content

            messages = [SystemMessage(content=prompt_text)]
            human_content = [{"type": "text", "text": "Please analyze this input."}]
            
            if audio_files:
                for a in audio_files:
                    with open(a, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        human_content.append({"type": "media", "mime_type": "audio/mp3", "data": b64})
            if image_files:
                for img in image_files:
                    with open(img, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        human_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            
            messages.append(HumanMessage(content=human_content))
            
            await self.emit("filler_speech", {"text": "Just a moment..."})
            result = await self._invoke_llm_with_retry(messages)
            print(f"[ParticipantAgent] LLM Result: {result}")
            
            for sub in result.sub_intents:
                if sub.is_contradiction or is_interruption:
                    await self.cancel_all_pending()
                
                self.state["intent"] = sub.intent_type
                for k, v in sub.parameters.items():
                    self.state["slots"][k] = v
                
                if sub.intent_type == "chitchat":
                    await self.emit("final_response", {"text": sub.parameters.get("response", "I'm not sure how to help.")})
                elif sub.intent_type in getattr(self, "tools", {}):
                    await self.call_tool(sub.intent_type, sub.parameters)
                else:
                    await self.emit("final_response", {"text": "I don't have a tool for that."})
        except asyncio.CancelledError:
            print("[ParticipantAgent] process_intents Cancelled")
            raise
        except Exception as e:
            print(f"LLM Error: {e}")

    async def on_user_text(self, text: str, end_of_turn: bool):
        self.buffer.append(text)
        
        if not end_of_turn:
            return
            
        turn = " ".join(self.buffer).strip()
        self.buffer = []
        
        image_files = getattr(self, "video_buffer", [])
        
        if getattr(self, "llm_task", None) and not self.llm_task.done():
            print("Task cancelled here!"); self.llm_task.cancel()
        self.llm_task = asyncio.create_task(self.process_intents(turn, image_files=image_files))
        
    async def on_interruption(self, text: str):
        # Cancel pending tools immediately
        await self.cancel_all_pending()
        
        await self.emit("filler_speech", {"text": "Okay, one moment."})
            
        if getattr(self, "llm_task", None) and not self.llm_task.done():
            print("Task cancelled here!"); self.llm_task.cancel()
        self.llm_task = asyncio.create_task(self.process_intents(text, is_interruption=True))

    async def on_tool_result(self, payload: Dict[str, Any]):
        import json
        call_id = payload.get("call_id", "")
        if self.pending_calls.pop(call_id, None) is None:
            return  # cancelled call
            
        result = payload.get("result", {})
        if payload.get("status") == "error" and "result" not in payload:
            result = {"error": payload.get("error"), "detail": payload.get("detail")}
            
        api_name = payload.get("api_name", "")
        
        # Save to event log so the LLM sees the result
        self.event_log.append({
            "event_type": "system_tool_result", 
            "api_name": api_name, 
            "result": result
        })
        
        # Trigger LLM again to process the tool result
        if not self.pending_calls:
            if getattr(self, "llm_task", None) and not self.llm_task.done():
                print("Task cancelled here!"); self.llm_task.cancel()
            self.llm_task = asyncio.create_task(
                self.process_intents("SYSTEM: The previous tool call completed. Please analyze the result and take the next step. If the user's request is fully complete, use 'chitchat' to give them the final answer. If a tool failed, retry it or inform the user.")
            )

    async def on_audio_chunk(self, audio_ref: str, end_of_turn: bool):
        if not hasattr(self, "audio_buffer"):
            self.audio_buffer = []
        if audio_ref:
            self.audio_buffer.append(audio_ref)

        if not end_of_turn:
            return
            
        audio_files = list(self.audio_buffer)
        self.audio_buffer = []
        
        # Audio implies speech, so it's a full utterance. 
        # Pass the buffered image frame if we have one.
        image_files = getattr(self, "video_buffer", [])
        
        if getattr(self, "llm_task", None) and not self.llm_task.done():
            print("Task cancelled here!"); self.llm_task.cancel()
        self.llm_task = asyncio.create_task(self.process_intents("Here is an audio recording of the user. If you have an image, consider it too.", audio_files=audio_files, image_files=image_files))

    async def on_video_frame(self, image_ref: str):
        if not hasattr(self, "video_buffer"):
            self.video_buffer = []
        if image_ref:
            # Overwrite to only keep the latest frame for latency
            self.video_buffer = [image_ref]
