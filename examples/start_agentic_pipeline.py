# import argparse

# from dacite import from_dict
# from hydra import compose, initialize
# from omegaconf import OmegaConf

# from roll.distributed.scheduler.initialize import init
# from roll.pipeline.agentic.agentic_config import AgenticConfig


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
#     parser.add_argument(
#         "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
#     )
#     args = parser.parse_args()

#     initialize(config_path=args.config_path, job_name="app")
#     cfg = compose(config_name=args.config_name)

#     print(OmegaConf.to_yaml(cfg, resolve=True))

#     ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

#     init()
#     from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

#     pipeline = AgenticPipeline(pipeline_config=ppo_config)

#     pipeline.run()


# if __name__ == "__main__":
#     main()


import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import json
import json
from typing import Any, Dict, List, Optional, Tuple
import uuid

import fleet
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from playwright.async_api import async_playwright

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class StepLog:
    step_number: int
    request: Optional[str]
    response_preview: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results_preview: List[str] = field(default_factory=list)
    success: bool = True
    ended: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)

class ComputerUseClient:
    def __init__(self, config: dict = None, debug: bool = False):
        self.config = config
        self.debug = debug

        self.available_tools = json.load(open("available_tools.json"))

        # if self.debug:
        #     print("[COMPUTER_USE] Number of tools: ", len(self.available_tools))

    async def connect(self, max_retries: int = 3) -> bool:
        """Initialize browser connection with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.debug:
                    print(f"[COMPUTER_USE] Browser connection attempt {attempt + 1}/{max_retries}")
                    
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,  # Always run in headless mode
                    args=[
                        '--no-sandbox', 
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                
                # Create page with realistic browser context
                context = await self.browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                )
                
                self.context = context
                self.page = await context.new_page()
                if self.debug:
                    print("[COMPUTER_USE] Browser initialized successfully")
                return True
                
            except Exception as e:
                if self.debug:
                    print(f"[COMPUTER_USE] Browser initialization failed (attempt {attempt + 1}): {e}")
                    
                # Clean up any partial state
                await self._cleanup_partial_state()
                
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    wait_time = 2 ** attempt
                    if self.debug:
                        print(f"[COMPUTER_USE] Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
        if self.debug:
            print(f"[COMPUTER_USE] Browser initialization failed after {max_retries} attempts")
        return False

    async def _cleanup_partial_state(self):
        """Clean up any partially initialized browser state."""
        try:
            if self.page:
                await self.page.close()
                self.page = None
            if self.context:
                await self.context.close()
                self.context = None
            if self.browser:
                await self.browser.close()
                self.browser = None
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            if self.debug:
                print(f"[COMPUTER_USE] Error during partial cleanup: {e}")

    async def take_screenshot(self) -> str:
        """Take screenshot with forced dimensions."""
        # Get the currently active page
        active_page = self.page
        if not active_page:
            if self.debug:
                print("[COMPUTER_USE] Cannot take screenshot: no active page")
            return ""

        try:
            if self.debug:
                print(f"[COMPUTER_USE] Setting viewport size to {self.config.policy.screenshot_width}x{self.config.policy.screenshot_height}")
            # Set viewport size for consistent screenshot dimensions
            await active_page.set_viewport_size({
                "width": 1400,
                "height": 850
            })

            # Take screenshot and return as base64
            screenshot_bytes = await active_page.screenshot(full_page=False)
            
            # Validate screenshot
            if not screenshot_bytes or len(screenshot_bytes) == 0:
                if self.debug:
                    print("[COMPUTER_USE] Screenshot is empty")
                return ""
                
            # Check size limit (5MB)
            if len(screenshot_bytes) > 5000000:
                if self.debug:
                    print(f"[COMPUTER_USE] Screenshot too large: {len(screenshot_bytes)} bytes")
                return ""
            
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Validate base64 encoding
            if not screenshot_b64:
                if self.debug:
                    print("[COMPUTER_USE] Base64 encoding failed")
                return ""
            
            # Return as data URI
            screenshot_data_uri = f"data:image/png;base64,{screenshot_b64}"
                
            if self.debug:
                print(f"[COMPUTER_USE] Screenshot taken successfully (bytes: {len(screenshot_bytes)}, b64: {len(screenshot_b64)} chars)")
            return screenshot_data_uri
        except Exception as e:
            if self.debug:
                print(f"[COMPUTER_USE] Could not take screenshot: {e}")
                import traceback
                print(f"[COMPUTER_USE] Screenshot error traceback: {traceback.format_exc()}")
            return ""


    def get_tools(self):
        return self.available_tools

    def call_tool(self, tool_name: str, arguments: dict):
        return self.available_tools[tool_name](arguments)


class ComputerUseRunner:
    def __init__(self, config: dict = None, debug: bool = False):
        self.config = config
        self.debug = debug
        self.client = ComputerUseClient(config=config, debug=debug)

        self.conversation_history_log = []
        self.screenshot_log = []
        self.issues_log = []
        self.model_self_assessment = None
        self.verifier_result = None

    def create_environment(self):
        env = fleet.env.make("booking", region="staging")
        instance_id = env.instance_id
        print(f"Instance ID: {instance_id}")

        instance_url = env.urls.app
        if isinstance(instance_url, list):
            instance_url = instance_url[0] if instance_url else ""
        return env, instance_url

    async def run_task(self, task: fleet.Task):
        # Create environment
        env, instance_url = self.create_environment()
        print(f"Instance URL: {instance_url}")
        if self.debug:
            print("[COMPUTER_USE] Sleeping for 10 seconds...")
        await asyncio.sleep(3)

        # computer_mcp = instance_url + "api/v1/mcp"
        # print(f"Computer MCP: {computer_mcp}")
        # async with streamablehttp_client(url=computer_mcp) as streams:
        #     async with ClientSession(
        #         read_stream=streams[0],
        #         write_stream=streams[1]
        #     ) as session:
        #         await session.initialize()
        #         print(f"[COMPUTER_USE] Session: {session}")
        #         tools_result = await session.list_tools()
        #         tools = tools_result.tools if hasattr(tools_result, 'tools') else []
        #         print(f"[COMPUTER_USE] Found {len(tools)} tools:\n")
        #         for i, tool in enumerate(tools, 1):
        #             print(f"[COMPUTER_USE] {i}. {tool}")
                
        #         print("--------------------------------\n\n")
        #         # result = await session.call_tool("computer", {"action": "screenshot"})


        #         input("Calling left_click tool...")
        #         result = await session.call_tool("computer", {"action": "left_click", "x": 100, "y": 100})
        #         print(f"[COMPUTER_USE] Result: {result}")

        # exit(0)

        # Connect to browser
        if self.debug:
            print("[COMPUTER_USE] Connecting to browser...")
        connected = await self.client.connect(max_retries=3)
        if not connected:
            error_msg = "Failed to initialize browser after retries"
            if self.debug:
                print(f"[ERROR] {error_msg}")
            return task.key, False, error_msg

        # Navigate to environment URL
        if self.debug:
            print(f"[COMPUTER_USE] Navigating to environment URL: {instance_url}")
        try:
            await self.client.page.goto(instance_url, wait_until="networkidle", timeout=60000)
            page_content = await self.client.page.content()

            if "502 Bad Gateway" in page_content or "Bad Gateway" in page_content:
                # Check if this is a real 502 error page vs just error text in a working page
                content_length = len(page_content.strip())
                error_appears_significant = (
                    content_length < 5000
                )  # Short content suggests error page

                if error_appears_significant and self.debug:
                    print(
                        f"[COMPUTER_USE] Detected significant Bad Gateway (content length: {content_length}), waiting 30 more seconds and retrying..."
                    )
                elif self.debug:
                    print(
                        f"[COMPUTER_USE] Bad Gateway text found but page has substantial content ({content_length} chars), continuing..."
                    )

                if error_appears_significant:
                    await asyncio.sleep(30)
                    try:
                        await self.client.page.reload(
                            wait_until="networkidle", timeout=90000
                        )
                        await asyncio.sleep(5)  # Additional short wait after reload
                        page_content = await self.client.page.content()
                        content_length = len(page_content.strip())

                        if (
                            "502 Bad Gateway" in page_content
                            or "Bad Gateway" in page_content
                        ) and content_length < 5000:
                            if self.debug:
                                print(
                                    "[COMPUTER_USE] Still getting significant Bad Gateway, trying one more time with fresh navigation..."
                                )
                            # Try fresh navigation instead of reload
                            current_url = self.client.page.url
                            await self.client.page.goto(
                                current_url, wait_until="networkidle", timeout=90000
                            )
                            await asyncio.sleep(10)
                            page_content = await self.client.page.content()
                            content_length = len(page_content.strip())

                            if (
                                "502 Bad Gateway" in page_content
                                or "Bad Gateway" in page_content
                            ) and content_length < 5000:
                                error_msg = f"Environment is returning Bad Gateway errors even after multiple retries (content length: {content_length})"
                                if self.debug:
                                    print(f"[ERROR] {error_msg}")
                                return task.key, False, error_msg
                    except Exception as retry_error:
                        error_msg = (
                            f"Failed during retry attempts: {str(retry_error)}"
                        )
                        if self.debug:
                            print(f"[ERROR] {error_msg}")
                        return task.key, False, error_msg
        except Exception as e:
            if self.debug:
                print(f"[COMPUTER_USE] Error navigating to environment URL: {e}")
            return False, f"Error navigating to environment URL: {e}"
        if self.debug:
            print("[COMPUTER_USE] Navigation completed successfully")

        run_id = str(uuid.uuid4())
        messages: List[Dict[str, Any]] = []
        available_tools = self.client.available_tools
        steps: List[StepLog] = []
        api_calls_log: List[Dict[str, Any]] = []

        # Build complete tool descriptions dynamically
        system_prompt = "test"
        messages.append({"role": "system", "content": system_prompt})

        # Log initial user message
        self.conversation_history_log.append(
            {
                "step": 0,
                "timestamp": datetime.now().isoformat(),
                "action": "system_prompt",
                "new_message": {"role": "system", "content": system_prompt},
            }
        )

        step_no = 0
        trajectory_ended = False
        success_overall = True
        error_reason: Optional[str] = None
        final_answer: Optional[str] = None
        task_completed_explicitly = False

        while not trajectory_ended and (
            self.config.policy.max_steps is None
            or step_no < self.config.policy.max_steps
        ):
            api_messages = self._limit_conversation_context(messages, step_no)

        return True, "Successfully navigated to environment URL"


async def main():
    computer_use_runner = ComputerUseRunner(debug=True)

    task = fleet.Task(
        key="test_key",
        prompt="Navigate to the environment URL and return the page content",
        env_id = "booking",
    )

    await computer_use_runner.run_task(task)
    exit(0)


# async def main():
#     env = fleet.env.make("booking", region="staging")
#     instance_id = env.instance_id
#     print(f"Instance ID: {instance_id}")

#     instance_url = env.urls.app
#     if isinstance(instance_url, list):
#         instance_url = instance_url[0] if instance_url else ""
    
#     print(f"[COMPUTER_USE] Instance URL: {instance_url}")

#     await asyncio.sleep(10)
#     input("Press Enter to continue...")

#     computer_mcp = instance_url + "api/v1/mcp"
#     print(f"Computer MCP: {computer_mcp}")

#     async with streamablehttp_client(url=computer_mcp) as streams:
#         async with ClientSession(
#             read_stream=streams[0],
#             write_stream=streams[1]
#         ) as session:
#             await session.initialize()
#             print(f"[COMPUTER_USE] Session: {session}")
#             tools_result = await session.list_tools()
#             tools = tools_result.tools if hasattr(tools_result, 'tools') else []
#             print(f"[COMPUTER_USE] Found {len(tools)} tools:\n")
#             for i, tool in enumerate(tools, 1):
#                 print(f"[COMPUTER_USE] {i}. {tool.name}")

#     # Clean up environment
#     input("Press Enter to close the environment...")
#     env.close()


if __name__ == "__main__":

    # tasks = fleet.load_tasks(env_key="fira")

    # task = tasks[0]

    # task.make(region="staging")

    # print(f"Tasks: {tasks}")
    asyncio.run(main())