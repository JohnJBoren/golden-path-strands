"""Ollama Provider for GPT-OSS and other models"""

import asyncio
import aiohttp
import json
from typing import Dict, Optional, Any, List
import structlog

logger = structlog.get_logger()


class OllamaProvider:
    """Provider for Ollama models including GPT-OSS:20b"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "gpt-oss:20b"):
        self.host = host.rstrip('/')
        self.model = model
        self.api_generate = f"{self.host}/api/generate"
        self.api_chat = f"{self.host}/api/chat"
        self.api_tags = f"{self.host}/api/tags"
        
    async def check_model_availability(self) -> bool:
        """Check if the specified model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_tags) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return self.model in models
                    return False
        except Exception as e:
            logger.error("model_check_failed", error=str(e))
            return False
    
    async def pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            async with aiohttp.ClientSession() as session:
                pull_url = f"{self.host}/api/pull"
                payload = {"name": self.model, "stream": False}
                
                logger.info("pulling_model", model=self.model)
                async with session.post(pull_url, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("model_pulled", model=self.model, status=result.get("status"))
                        return True
                    return False
        except Exception as e:
            logger.error("model_pull_failed", error=str(e))
            return False
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response using Ollama model"""
        
        # Ensure model is available
        if not await self.check_model_availability():
            logger.info("model_not_found_pulling", model=self.model)
            if not await self.pull_model():
                raise RuntimeError(f"Failed to pull model {self.model}")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_generate, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "response": result.get("response", ""),
                            "total_tokens": result.get("total_duration", 0) // 1000000,  # Convert ns to ms
                            "model": self.model,
                            "done": result.get("done", True),
                        }
                    else:
                        error_text = await response.text()
                        logger.error("ollama_error", status=response.status, error=error_text)
                        return {"response": "", "error": error_text}
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            return {"response": "", "error": str(e)}
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Chat completion using Ollama model"""
        
        # Ensure model is available
        if not await self.check_model_availability():
            logger.info("model_not_found_pulling", model=self.model)
            if not await self.pull_model():
                raise RuntimeError(f"Failed to pull model {self.model}")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_chat,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "response": result.get("message", {}).get("content", ""),
                            "role": result.get("message", {}).get("role", "assistant"),
                            "total_tokens": result.get("total_duration", 0) // 1000000,
                            "model": self.model,
                            "done": result.get("done", True),
                        }
                    else:
                        error_text = await response.text()
                        logger.error("ollama_chat_error", status=response.status, error=error_text)
                        return {"response": "", "error": error_text}
        except Exception as e:
            logger.error("chat_failed", error=str(e))
            return {"response": "", "error": str(e)}
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate response with tool usage capability"""
        
        tool_descriptions = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools
        ])
        
        enhanced_prompt = f"""
You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
TOOL: <tool_name>
ARGS: <arguments>

Task: {prompt}

Please complete the task using the available tools when necessary.
"""
        
        response = await self.generate(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Parse tool usage from response
        response_text = response.get("response", "")
        tools_used = []
        
        if "TOOL:" in response_text:
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                    args = ""
                    if i + 1 < len(lines) and lines[i + 1].startswith("ARGS:"):
                        args = lines[i + 1].replace("ARGS:", "").strip()
                    tools_used.append({"name": tool_name, "args": args})
        
        response["tools"] = tools_used
        response["reasoning"] = self._extract_reasoning(response_text)
        
        return response
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from response text"""
        reasoning_markers = ["Reasoning:", "Thinking:", "Analysis:", "Step"]
        reasoning = []
        
        lines = text.split('\n')
        capture = False
        
        for line in lines:
            if any(marker in line for marker in reasoning_markers):
                capture = True
            if capture and line.strip():
                reasoning.append(line)
            if "TOOL:" in line or "Result:" in line:
                capture = False
        
        return "\n".join(reasoning)