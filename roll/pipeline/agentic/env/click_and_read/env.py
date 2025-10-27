import numpy as np
from PIL import Image
from typing import Tuple, Any, Dict
import random
import logging
import fleet
import json
import base64
import asyncio
from playwright.async_api import async_playwright


from gem import Env
from roll.pipeline.agentic.utils import all_seed

logger = logging.getLogger(__name__)


TOOLS_DESCRIPTION = """1. browser - Interact with the web page using the browser tool with an action parameter:
   - browser(action="left_click", x=X, y=Y) - Click on elements at specific coordinates
   - browser(action="right_click", x=X, y=Y) - Right-click on elements
   - browser(action="double_click", x=X, y=Y) - Double-click on elements
   - browser(action="type", text="your text") - Type text into focused elements
   - browser(action="key", text="Enter") - Press keyboard keys (Enter, Escape, Tab, etc.)
   - browser(action="scroll", x=X, y=Y, scroll_direction="down", scroll_amount=3) - Scroll pages
   - browser(action="wait", duration=2) - Wait for page updates
   - browser(action="left_click_drag", start_x=X1, start_y=Y1, x=X2, y=Y2) - Drag elements

Examples:
- To click a button: browser(action="left_click", x=352, y=93)
- To type text: browser(action="type", text="hello world")
- To press Enter: browser(action="key", text="Enter")

2. complete_task - Signal when you have successfully completed the task:
   - complete_task(success=true/false, summary="description of what was accomplished")
   - complete_task(success=true, summary="Created Jira ticket", answer="OEP-130") - Include answer if task asks for specific data

3. give_up - LAST RESORT: Give up on the task when you have absolutely no other alternatives:
   - give_up(reason="detailed explanation", attempts_made=["list", "of", "specific attempts"])
   - Only use when: 1) All approaches exhausted, 2) Task appears impossible, 3) Stuck in unrecoverable state"""

def get_system_prompt(task: fleet.Task):
    return f"""You are an AI assistant that can interact with web browsers to solve tasks.

    Your task: {task.prompt}

    You have access to the following tools:

    {TOOLS_DESCRIPTION}

    Important guidelines:
    - Be methodical and break down complex tasks into steps
    - Use coordinates carefully - they should be within the screen bounds
    - Common workflow: click element → type text → press Enter"""

class ComputerUseClient:
    def __init__(self, config: dict = None, debug: bool = False):
        self.config = config
        self.debug = debug
        self.available_tools = json.load(open("available_tools.json"))
        self.last_action = None
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

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

    async def take_screenshot(self) -> np.ndarray:
        """Take screenshot and return as numpy array (H, W, 3) RGB."""
        # Get the currently active page
        active_page = self.page
        if not active_page:
            if self.debug:
                print("[COMPUTER_USE] Cannot take screenshot: no active page")
            return np.zeros((850, 1400, 3), dtype=np.uint8)

        try:
            if self.debug:
                print(f"[COMPUTER_USE] Setting viewport size to {self.config.policy.screenshot_width}x{self.config.policy.screenshot_height}")
            # Set viewport size for consistent screenshot dimensions
            await active_page.set_viewport_size({
                "width": 1400,
                "height": 850
            })

            # Take screenshot
            screenshot_bytes = await active_page.screenshot(full_page=False)
            
            # Validate screenshot
            if not screenshot_bytes or len(screenshot_bytes) == 0:
                if self.debug:
                    print("[COMPUTER_USE] Screenshot is empty")
                return np.zeros((850, 1400, 3), dtype=np.uint8)

            # Check size limit (5MB)
            if len(screenshot_bytes) > 5000000:
                if self.debug:
                    print(f"[COMPUTER_USE] Screenshot too large: {len(screenshot_bytes)} bytes")
                return np.zeros((850, 1400, 3), dtype=np.uint8)

            from io import BytesIO
            image = Image.open(BytesIO(screenshot_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)

            if self.debug:
                print(f"[COMPUTER_USE] Screenshot taken successfully: shape={image_array.shape}, dtype={image_array.dtype}")

            return image_array

        except Exception as e:
            if self.debug:
                print(f"[COMPUTER_USE] Could not take screenshot: {e}")
                import traceback
                print(f"[COMPUTER_USE] Screenshot error traceback: {traceback.format_exc()}")
            return np.zeros((850, 1400, 3), dtype=np.uint8)

    def _get_active_page(self):
        """Get the currently active page."""
        return self.page

    async def _is_page_connected(self, page) -> bool:
        """Check if a page is still connected and responsive."""
        try:
            if not page:
                return False
            # Simple check - try to get the page title
            await page.title()
            return True
        except Exception:
            return False

    async def _reconnect_browser(self) -> bool:
        """Attempt to reconnect the browser after connection loss."""
        if self.debug:
            print("[COMPUTER_USE] Attempting browser reconnection...")
        
        # Clean up existing state
        await self._cleanup_partial_state()
        
        # Try to reconnect
        return await self.connect(max_retries=2)

    async def _execute_browser_action(self, action_params: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """Execute a browser action using Playwright with retry logic."""
        active_page = self._get_active_page()
        if not active_page:
            return {"success": False, "error": "Browser not initialized"}
        
        action = action_params.get("action")
        
        if self.debug:
            print(f"[COMPUTER_USE] Browser Action: {action}({action_params})")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0 and self.debug:
                    print(f"[COMPUTER_USE] Retrying browser action (attempt {attempt + 1}/{max_retries})")
                    
                # Check if page is still connected
                if not await self._is_page_connected(active_page):
                    if self.debug:
                        print("[COMPUTER_USE] Page disconnected, attempting reconnection")
                    if not await self._reconnect_browser():
                        return {"success": False, "error": "Browser connection lost and reconnection failed"}
                    active_page = self._get_active_page()
                    if not active_page:
                        return {"success": False, "error": "Failed to get active page after reconnection"}
                
                # Execute the browser action
                if action == "left_click":
                    x = action_params.get("x", 0)
                    y = action_params.get("y", 0)
                    await active_page.mouse.click(x, y)
                    await asyncio.sleep(0.2)  # Small delay for click to register
                    self.last_action = {"type": "click", "target": {"x": x, "y": y}}
                    return {"success": True}
                
                elif action == "type":
                    text = action_params.get("text", "")
                    await active_page.keyboard.type(text)
                    self.last_action = {"type": "type", "text": text}
                    return {"success": True}
                
                elif action == "key":
                    key = action_params.get("text", "")  # Use text field for key name
                    # Map common key names to Playwright-compatible names
                    key_mapping = {
                        "escape": "Escape",
                        "enter": "Enter", 
                        "return": "Enter",
                        "tab": "Tab",
                        "space": "Space",
                        "backspace": "Backspace",
                        "delete": "Delete",
                        "up": "ArrowUp",
                        "down": "ArrowDown", 
                        "left": "ArrowLeft",
                        "right": "ArrowRight",
                        "home": "Home",
                        "end": "End",
                        "pageup": "PageUp",
                        "pagedown": "PageDown",
                        "ctrl": "Control",
                        "alt": "Alt",
                        "shift": "Shift",
                        "cmd": "Meta",
                        "meta": "Meta"
                    }
                    mapped_key = key_mapping.get(key.lower(), key)
                    await active_page.keyboard.press(mapped_key)
                    self.last_action = {"type": "key", "key": mapped_key}
                    return {"success": True}
                
                elif action == "scroll":
                    x = action_params.get("x", 0)
                    y = action_params.get("y", 0)
                    direction = action_params.get("scroll_direction", "down")
                    amount = action_params.get("scroll_amount", 3)

                    # Convert direction and amount to delta values for Playwright
                    # Default multiplier if config not available
                    multiplier = 100
                    if self.config and hasattr(self.config, 'policy') and hasattr(self.config.policy, 'scroll_amount_multiplier'):
                        multiplier = self.config.policy.scroll_amount_multiplier
                    
                    delta_x = 0
                    delta_y = 0

                    if direction == "down":
                        delta_y = amount * multiplier
                    elif direction == "up":
                        delta_y = -amount * multiplier
                    elif direction == "right":
                        delta_x = amount * multiplier
                    elif direction == "left":
                        delta_x = -amount * multiplier

                    # Move mouse to position and scroll
                    await active_page.mouse.move(x, y)
                    await active_page.mouse.wheel(delta_x, delta_y)
                    self.last_action = {"type": "scroll", "direction": direction}
                    return {"success": True}
                
                elif action == "wait":
                    duration = action_params.get("duration", 1)
                    await asyncio.sleep(duration)
                    self.last_action = {"type": "wait", "duration": duration}
                    return {"success": True}
                
                elif action in ["right_click", "double_click"]:
                    x = action_params.get("x", 0)
                    y = action_params.get("y", 0)

                    if action == "right_click":
                        await active_page.mouse.click(x, y, button="right")
                    elif action == "double_click":
                        await active_page.mouse.dblclick(x, y)

                    self.last_action = {"type": action, "target": {"x": x, "y": y}}
                    return {"success": True}
                
                elif action == "left_click_drag":
                    start_x = action_params.get("start_x", 0)
                    start_y = action_params.get("start_y", 0)
                    end_x = action_params.get("x", 0)
                    end_y = action_params.get("y", 0)

                    # Perform drag using mouse down, move, up sequence
                    await active_page.mouse.move(start_x, start_y)
                    await active_page.mouse.down()
                    await active_page.mouse.move(end_x, end_y)
                    await active_page.mouse.up()

                    self.last_action = {"type": "drag", "from": {"x": start_x, "y": start_y}, "to": {"x": end_x, "y": end_y}}
                    return {"success": True}
            
                else:
                    return {"success": False, "error": f"Unknown browser action: {action}"}
                    
            except Exception as e:
                if "Connection closed" in str(e) or "Target page, context or browser has been closed" in str(e):
                    if self.debug:
                        print(f"[COMPUTER_USE] Connection error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        # Try to reconnect
                        if await self._reconnect_browser():
                            active_page = self._get_active_page()
                            continue
                    return {"success": False, "error": f"Browser connection lost: {str(e)}"}
                elif attempt < max_retries - 1:
                    if self.debug:
                        print(f"[COMPUTER_USE] Browser action error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    if self.debug:
                        print(f"[COMPUTER_USE] Browser action failed after {max_retries} attempts: {e}")
                    return {"success": False, "error": f"Browser action failed: {str(e)}"}
        
        return {"success": False, "error": f"Browser action failed after {max_retries} attempts"}

class ComputerUseRunner:
    def __init__(self, task: fleet.Task, env, config: dict = None, debug: bool = False):
        self.task = task
        self.env = env

        instance_url = env.urls.app
        if isinstance(instance_url, list):
            instance_url = instance_url[0] if instance_url else ""
        self.instance_url = instance_url

        self.config = config
        self.debug = debug
        self.client = ComputerUseClient(config=config, debug=debug)

        self.conversation_history_log = []
        self.screenshot_log = []
        self.issues_log = []
        self.model_self_assessment = None
        self.verifier_result = None

    async def connect(self) -> Tuple[str, bool, str]:

        task = self.task
        instance_url = self.instance_url

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
            return task.key, False, f"Error navigating to environment URL: {e}"
        if self.debug:
            print("[COMPUTER_USE] Navigation completed successfully")
        
        return task.key, True, None


class ClickAndReadEnv(Env):
    """
    Browser automation environment using Fleet tasks.
    
    The agent interacts with web pages through browser actions (click, type, scroll, etc.)
    to complete tasks provided by Fleet. Screenshots are returned as observations,
    and tasks are verified through Fleet's verification system.
    
    Args:
        tasks: List of Fleet tasks to use for episodes
        max_steps: Maximum number of actions per episode
        image_size: Not used (kept for compatibility)
        button_size: Not used (kept for compatibility)
        format_penalty: Penalty for invalid action format
        render_mode: Rendering mode (default: "rgb_array")
    """

    def __init__(
        self,
        max_steps=5,
        image_size=100,
        button_size=25,
        format_penalty=-0.01,
        render_mode="rgb_array",
        **kwargs
    ):
        super().__init__()
        self.max_steps = max_steps
        self.image_size = image_size
        self.button_size = button_size
        self.format_penalty = format_penalty
        self.render_mode = render_mode
        self.tasks = fleet.load_tasks(env_key="booking")
        self.selected_task = None
        self.env = None
        self.client = None

        # 4 fixed button positions (corners)
        # margin = 0
        # self.BUTTON_POSITIONS = {
        #     "top_left": (margin, margin),
        #     "top_right": (image_size - button_size - margin, margin),
        #     "bottom_left": (margin, image_size - button_size - margin),
        #     "bottom_right": (image_size - button_size - margin, image_size - button_size - margin),
        # }
        
        # # Word bank for the task
        # self.WORDS = ["APPLE", "CHERRY", "EAGLE", "FOREST", "HELLO"]

        # State tracking
        self.current_step = 0
        # self.button_clicked = False
        # self.button_position = None  # (x, y) of top-left corner
        # self.button_location_name = None  # "top_left", etc.
        # self.target_word = None
        self.current_image = None

    def get_instructions(self) -> str:
        """Generate task-specific instructions
        
        This is returned as env_instruction in the info dict and gets automatically
        prepended to the first user message at step 0 by VLTrajEnvManager.
        
        Note: Tool descriptions are in the system prompt, so we only include the
        specific task here to avoid duplication.
        """
        if self.selected_task:
            task_prompt = self.selected_task.prompt
        else:
            task_prompt = "Task not yet initialized"
        
        return f"""TASK: {task_prompt}

Analyze the screenshot carefully and use the available tools to complete this task step by step.
"""

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        Env.reset(self, seed)

        # Reset state
        self.current_step = 0

        with all_seed(seed):
            # Get a random task from fleet and create an instance
            self.selected_task = random.choice(self.tasks)
            self.env = self.selected_task.make(region="staging")
            self.runner = ComputerUseRunner(task=self.selected_task, env=self.env, config=None, debug=True)
            _, success, error_msg = asyncio.run(self.runner.connect())
            self.success = success
            if not success:
                raise ValueError(f"Failed to connect to browser: {error_msg}")

            self.current_image = asyncio.run(self.runner.client.take_screenshot())

        return self.current_image, {"env_instruction": self.get_instructions()}

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment
        
        Args:
            action: LLM output string containing tool calls
            
        Returns:
            observation: Screenshot of the current state (numpy array)
            reward: Reward from task verification
            terminated: Whether episode is done
            truncated: Whether episode was truncated (max steps)
            info: Additional information dictionary
        """
        self.current_step += 1
        
        # Parse action to extract tool calls
        action_info = self.parse_action(action)
        tool_name = action_info["tool_name"]
        action_is_valid = action_info["action_is_valid"]
        tool_params = action_info.get("params", {})

        logger.debug(
            f"Step {self.current_step} - Tool: {tool_name}, "
            f"Valid: {action_is_valid}, Params: {tool_params}"
        )
        
        reward = 0.0
        terminated = False
        execution_error = None
        
        # Execute the tool
        if not action_is_valid:
            # Invalid format - apply penalty
            reward = self.format_penalty
            execution_error = "Invalid tool format"
            logger.debug(f"Invalid action format - penalty: {reward}")
            
        elif tool_name == "browser":
            # Execute browser action
            try:
                result = asyncio.run(self.runner.client._execute_browser_action(tool_params))
                if not result.get("success"):
                    execution_error = result.get("error", "Browser action failed")
                    logger.debug(f"Browser action failed: {execution_error}")
            except Exception as e:
                execution_error = f"Browser action exception: {str(e)}"
                logger.error(f"Browser action exception: {e}")
                
        elif tool_name == "complete_task":
            # Task completion signaled by agent
            terminated = True
            success = tool_params.get("success", False)
            summary = tool_params.get("summary", "")
            answer = tool_params.get("answer", None)
            
            logger.debug(f"Agent signaled task completion: success={success}, summary={summary}, answer={answer}")
            
            # Verify the task (sync method)
            try:
                reward = self.selected_task.verify(self.env)
                logger.debug(f"Task verification reward: {reward}")
            except Exception as e:
                logger.error(f"Task verification failed: {e}")
                reward = 0.0
                
        elif tool_name == "give_up":
            # Agent gave up
            terminated = True
            reason = tool_params.get("reason", "No reason provided")
            attempts = tool_params.get("attempts_made", [])
            logger.debug(f"Agent gave up: {reason}, attempts: {attempts}")
            reward = 0.0  # No reward for giving up
            
        else:
            # Unknown tool
            execution_error = f"Unknown tool: {tool_name}"
            reward = self.format_penalty
            logger.debug(execution_error)
        
        # Take screenshot of current state (after action execution)
        try:
            self.current_image = asyncio.run(self.runner.client.take_screenshot())
        except Exception as e:
            logger.error(f"Failed to take screenshot after action: {e}")
            # Keep previous image if screenshot fails
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            if not terminated:
                logger.debug(f"Episode truncated - max steps ({self.max_steps}) reached")
            truncated = True
            terminated = True
        else:
            truncated = False
        
        # Metrics
        metrics = {
            "action_is_valid": action_is_valid,
            "success": reward > 0,
            "steps_used": self.current_step,
            "tool_name": tool_name,
            "execution_error": execution_error is not None,
        }
        
        if execution_error:
            metrics["error_message"] = execution_error
        
        info = {
            "metrics": metrics,
            "metrics_agg_mode": {
                "action_is_valid": "mean",
                "success": "last",
                "steps_used": "mean",
                "execution_error": "sum",
            },
        }
        
        return self.current_image, reward, terminated, truncated, info

    def _is_click_on_button(self, x: int, y: int) -> bool:
        """Check if click coordinates are within the button bounds"""
        bx, by = self.button_position
        return (bx <= x <= bx + self.button_size and 
                by <= y <= by + self.button_size)

    def parse_action(self, text: str) -> Dict:
        """Parse tool calls from LLM output
        
        Expected formats:
        - browser(action="left_click", x=100, y=200)
        - complete_task(success=true, summary="Task done", answer="result")
        - give_up(reason="Cannot proceed", attempts_made=["attempt1", "attempt2"])
        """
        import re
        
        # Remove chat tokens
        for token in ("<|im_start|>", "<|im_end|>"):
            text = text.replace(token, "")
        
        text = text.strip()
        
        # Try to parse browser tool
        browser_match = re.search(
            r'browser\s*\(\s*action\s*=\s*["\']([^"\']+)["\']([^)]*)\)',
            text,
            re.IGNORECASE
        )
        if browser_match:
            action_name = browser_match.group(1)
            params_str = browser_match.group(2)
            
            params = {"action": action_name}
            
            # Parse additional parameters
            # x=100, y=200
            for param_match in re.finditer(r'(\w+)\s*=\s*([^,\)]+)', params_str):
                key = param_match.group(1).strip()
                value = param_match.group(2).strip().strip('"\'')
                
                # Try to convert to int
                try:
                    params[key] = int(value)
                except ValueError:
                    params[key] = value
            
            return {
                "tool_name": "browser",
                "action_is_valid": True,
                "params": params,
            }
        
        # Try to parse complete_task tool
        complete_match = re.search(
            r'complete_task\s*\(([^)]+)\)',
            text,
            re.IGNORECASE
        )
        if complete_match:
            params_str = complete_match.group(1)
            params = {}
            
            # Parse success parameter
            success_match = re.search(r'success\s*=\s*(true|false)', params_str, re.IGNORECASE)
            if success_match:
                params["success"] = success_match.group(1).lower() == "true"
            
            # Parse summary parameter
            summary_match = re.search(r'summary\s*=\s*["\']([^"\']+)["\']', params_str)
            if summary_match:
                params["summary"] = summary_match.group(1)
            
            # Parse answer parameter (optional)
            answer_match = re.search(r'answer\s*=\s*["\']([^"\']+)["\']', params_str)
            if answer_match:
                params["answer"] = answer_match.group(1)
            
            return {
                "tool_name": "complete_task",
                "action_is_valid": True,
                "params": params,
            }
        
        # Try to parse give_up tool
        giveup_match = re.search(
            r'give_up\s*\(([^)]+)\)',
            text,
            re.IGNORECASE
        )
        if giveup_match:
            params_str = giveup_match.group(1)
            params = {}
            
            # Parse reason parameter
            reason_match = re.search(r'reason\s*=\s*["\']([^"\']+)["\']', params_str)
            if reason_match:
                params["reason"] = reason_match.group(1)
            
            # Parse attempts_made (simplified - just extract the string)
            attempts_match = re.search(r'attempts_made\s*=\s*\[([^\]]+)\]', params_str)
            if attempts_match:
                attempts_str = attempts_match.group(1)
                # Split by comma and clean up quotes
                attempts = [a.strip().strip('"\'') for a in attempts_str.split(',')]
                params["attempts_made"] = attempts
            
            return {
                "tool_name": "give_up",
                "action_is_valid": True,
                "params": params,
            }
        
        # Invalid format - no recognized tool
        return {
            "tool_name": None,
            "action_is_valid": False,
            "params": {},
        }

