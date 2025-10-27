# MCPTool Flow in ROLL Rollout Process

This document explains how `MCPTool` is integrated and called during the rollout process in the ROLL framework.

## High-Level Overview

```
YAML Config → TrajEnvManager → ToolEnvWrapper → MCPTool
                     ↓
              Environment Step
                     ↓
         GEM ToolEnvWrapper intercepts action
                     ↓
            MCPTool.execute_action()
                     ↓
         MCP Server (you specify URL later)
```

## Detailed Flow

### 1. **Tool Registration** (Startup Time)

Location: `roll/pipeline/agentic/tools/__init__.py`

```python
from roll.pipeline.agentic.tools.registration import register_tools

# MCPTool is registered at module import time
register_tools(
    tool_id="mcp", 
    entry_point="roll.pipeline.agentic.tools.mcp_tool:MCPTool"
)
```

**Key Point**: MCPTool is registered with ID `"mcp"` in the global `TOOL_REGISTRY`.

---

### 2. **YAML Configuration** (You Define This)

Location: Your config file (e.g., `examples/config/traj_envs_gem_qa.yaml`)

```yaml
HotpotQA_with_mcp:
  env_type: "roll_qa"  # Your base environment
  max_steps: 10
  max_tokens_per_step: 128
  env_config:
    dataset_name: axon-rl/HotpotQA
    split: train
  
  # THIS IS WHERE MCP TOOL IS CONFIGURED
  tool_wrapper:
    wrapper_args:
      tool_reward: 0.0
      tool_success_reward: 0.2
      max_tool_uses: 1
    tool_configs:
      - tool_id: mcp  # References the registered tool
        tool_args:
          server_url: xxx  # ← You can set this at runtime!
```

**Important**: The `server_url` can be any placeholder (like `xxx`) because you can override it when you start the rollout by passing it as a runtime parameter or environment variable.

---

### 3. **Environment Manager Initialization** (Rollout Start)

Location: `roll/pipeline/agentic/env_manager/traj_env_manager.py:66-75`

```python
class TrajEnvManager(BaseEnvManager):
    def __init__(self, ...):
        # Step 1: Create the base environment
        self.env = gem.make(
            env_id=self.env_config["env_type"], 
            **self.env_config['config']
        )
        
        # Step 2: If tool_wrapper config exists, wrap the environment
        if "tool_wrapper" in self.env_config:
            self.env = tool_wrapper(
                self.env,
                wrapper_args=self.env_config.tool_wrapper.wrapper_args,
                tool_configs=self.env_config.tool_wrapper.tool_configs
            )
```

---

### 4. **Tool Instantiation**

Location: `roll/pipeline/agentic/tools/tool_env_wrapper.py:46-55`

```python
def tool_wrapper(env: Env, wrapper_args: Dict, tool_configs: List[Dict]):
    tools = []
    
    # For each tool in config
    for tool_config in tool_configs:
        tool_id = tool_config["tool_id"]  # "mcp"
        tool_args = tool_config["tool_args"]  # {"server_url": "xxx"}
        
        # Create the tool instance using the registry
        tools.append(make_tool(tool_id=tool_id, **tool_args))
    
    # Wrap the environment with the tools
    tool_env_wrapper = ToolEnvWrapper(env, tools, **wrapper_args)
    return tool_env_wrapper
```

**What happens here:**
- `make_tool(tool_id="mcp", server_url="xxx")` looks up `"mcp"` in `TOOL_REGISTRY`
- It calls `MCPTool(server_url="xxx")`
- The `MCPTool` instance is passed to `ToolEnvWrapper`

---

### 5. **MCPTool Initialization**

Location: `roll/pipeline/agentic/tools/mcp_tool.py:26-47`

```python
class MCPTool(BaseTool):
    def __init__(self, 
                 num_workers=1, 
                 server_url: Optional[str] = None, 
                 client: Optional[MCPClient] = None,
                 tool_names_subset: Optional[List[str]] = None,
                 custom_prompt: Optional[str] = None):
        super().__init__(num_workers)
        
        if not client and not server_url:
            raise ValueError("Either 'client' or 'server_url' must be provided.")
        
        # Create the MCP client (not connected yet!)
        self._client = client or MCPClient(server_url)
        self._is_connected_and_ready = False
        
        # Setup event loop for async operations
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
```

**Key Point**: The MCPTool doesn't connect to the server immediately. Connection happens lazily on first use.

---

### 6. **Environment Reset**

Location: `roll/pipeline/agentic/env_manager/traj_env_manager.py:153-171`

```python
def reset(self) -> RolloutCache:
    # ...
    observation, info = self.env.reset(seed=seed)
    # ...
```

**What happens:**
- `self.env` is the `ToolEnvWrapper` instance
- `ToolEnvWrapper.reset()` calls the wrapped environment's `reset()`
- It also calls `tool.instruction_string()` on each tool to inject tool instructions

Location: GEM's `tool_env_wrapper.py` (in GEM package)

```python
class ToolEnvWrapper(Env):
    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        
        # Get tool instructions and inject them
        tool_instructions = []
        for tool in self.tools:
            tool_instructions.append(tool.instruction_string())
        
        # Combine tool instructions with env instructions
        env_instruction = info.get("env_instruction", "")
        combined_instruction = env_instruction + "\n" + "\n".join(tool_instructions)
        info["env_instruction"] = combined_instruction
        
        return obs, info
```

**MCPTool's `instruction_string()` method** (lines 49-64):

```python
def instruction_string(self) -> str:
    """
    Returns the instruction string for the agent.
    Connects to MCP server if not already connected.
    """
    self._ensure_connected()  # ← CONNECTS HERE ON FIRST CALL
    
    if self._custom_prompt:
        return self._custom_prompt
    
    return self._generate_prompt_from_cached_tools()
```

**This is where the MCP server connection happens!** The first time `instruction_string()` is called:
1. `_ensure_connected()` checks if connected
2. If not, calls `_async_connect_and_fetch()` which:
   - Connects to the MCP server at `server_url`
   - Fetches available tools from the server
   - Caches tool metadata
3. Generates a prompt with tool instructions for the agent

---

### 7. **Rollout Loop - Agent Action**

Location: `roll/pipeline/agentic/env_manager/traj_env_manager.py:121-131`

```python
while self.running:
    # Agent generates an action
    lm_output: DataProto = self.make_decision(rollout_cache)
    
    # Environment processes the action
    rollout_cache: RolloutCache = self.step(lm_output)
```

---

### 8. **Environment Step with Tool Execution**

Location: `roll/pipeline/agentic/env_manager/traj_env_manager.py:173-205`

```python
def step(self, llm_output: DataProto):
    # Decode the agent's action
    responses = self.tokenizer.batch_decode(
        llm_output.batch['responses'], 
        skip_special_tokens=False
    )
    
    # Call the environment's step (which is ToolEnvWrapper)
    observation, reward, terminated, truncated, info = self.env.step(
        action=responses[0]
    )
    # ...
```

**What happens in `ToolEnvWrapper.step()`** (from GEM):

```python
class ToolEnvWrapper(Env):
    def step(self, action: str):
        # Try each tool to see if it can handle the action
        tool_executed = False
        observation = ""
        
        for tool in self.tools:
            # Each tool checks if the action is for it
            is_parsed, is_valid, obs, parsed_action = tool.execute_action(action)
            
            if is_parsed:
                # This tool handled the action
                tool_executed = True
                observation = obs
                self.tool_use_counter += 1
                if is_valid:
                    self.tool_success_counter += 1
                break
        
        # If no tool handled it, pass to the underlying environment
        if not tool_executed:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            # Tool was executed, but environment continues
            obs = observation
            reward = self.tool_reward if is_valid else 0
            terminated = False
            truncated = False
            info = {}
        
        # Return modified observation with tool results
        return obs, reward, terminated, truncated, info
```

---

### 9. **MCPTool Execution**

Location: `roll/pipeline/agentic/tools/mcp_tool.py:66-132`

```python
def execute_action(self, action: str) -> Tuple[bool, bool, str, str]:
    """
    Parses and executes a tool call from the agent's action string.
    
    Returns:
        (is_parsed, is_valid, observation, parsed_action)
    """
    self._ensure_connected()  # Ensure connected (usually already is)
    
    # Stage 1: Parse the action
    json_content, parsed_action, is_parsed = self._parse_action(action)
    
    if not is_parsed:
        # No <tool_call> tag found - not for this tool
        return (False, False, "", action)
    
    # Stage 2: Validate the JSON
    try:
        data = json.loads(json_content)
        tool_name = data.get("tool_name")
        tool_params = data.get("tool_params", {})
        
        # Validate against tool schema
        self._validate_tool_call(tool_name, tool_params)
        
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        error_msg = f"[Validation Error: {e}]"
        return (True, False, error_msg, parsed_action)
    
    # Stage 3: Execute the tool on the MCP server
    try:
        result = self._run_async_logic(
            self._client.call_tool(tool_name, tool_params)
        )
        
        is_success, observation = self._process_server_response(result)
        return (True, is_success, observation, parsed_action)
        
    except Exception as e:
        error_msg = f"[Execution Error: {e}]"
        return (True, False, error_msg, parsed_action)
```

**Agent Action Format:**

The agent produces actions like:
```
<tool_call>
{
  "tool_name": "search",
  "tool_params": {
    "query": "What is Python?"
  }
}
</tool_call>
```

**MCPTool flow:**
1. Parses the `<tool_call>` tag
2. Extracts JSON
3. Validates tool name and parameters against the schema
4. Calls the MCP server via `MCPClient.call_tool()`
5. Returns the result as an observation

---

### 10. **MCP Server Communication**

Location: `roll/pipeline/agentic/env/mcp/mcp_client.py`

```python
class MCPClient:
    async def call_tool(self, tool_name: str, arguments: Dict) -> types.CallToolResult:
        """Call a tool on the MCP server"""
        # Make HTTP/SSE request to MCP server
        result = await self.session.call_tool(
            name=tool_name,
            arguments=arguments
        )
        return result
```

The MCP server processes the request and returns a result.

---

## Summary of the Complete Flow

```
1. [Startup] MCPTool registered with ID "mcp"
2. [Config] YAML specifies tool_wrapper with mcp tool and server_url
3. [Init] TrajEnvManager creates environment
4. [Init] tool_wrapper() instantiates MCPTool with server_url
5. [Reset] ToolEnvWrapper calls MCPTool.instruction_string()
   └─→ MCPTool connects to MCP server
   └─→ Fetches available tools
   └─→ Generates prompt with tool instructions
6. [Loop] Agent generates action
7. [Step] ToolEnvWrapper.step(action)
   └─→ Tries each tool's execute_action()
   └─→ MCPTool.execute_action() checks for <tool_call> tag
   └─→ If found:
       ├─→ Parses JSON
       ├─→ Validates against schema
       ├─→ Calls MCP server
       └─→ Returns observation to agent
8. [Continue] Loop continues until episode terminates
```

---

## Regarding Your Question: "I don't know the server until I start the rollout"

**This is fine!** You have several options:

### Option 1: Environment Variable
Set the server URL via environment variable before running:

```bash
export MCP_SERVER_URL="http://localhost:8000"
python examples/start_agentic_pipeline.py --config your_config.yaml
```

Then in your YAML:
```yaml
tool_args:
  server_url: ${oc.env:MCP_SERVER_URL}
```

### Option 2: Command-Line Override
Use Hydra's command-line overrides:

```bash
python examples/start_agentic_pipeline.py \
  --config your_config.yaml \
  custom_envs.YourEnv.tool_wrapper.tool_configs.0.tool_args.server_url=http://localhost:8000
```

### Option 3: Runtime Configuration
Pass it programmatically when creating the environment in your custom script.

### Option 4: Lazy Connection (Current Behavior)
The `MCPTool` doesn't connect until the first `reset()` call, so you can:
1. Start the rollout with a placeholder URL
2. The actual connection happens when the environment resets
3. By that time, your MCP server should be running

---

## Key Insights

1. **Tools are wrappers around environments**: The `ToolEnvWrapper` intercepts actions before they reach the base environment

2. **Tools inspect actions**: Each tool checks if an action is meant for it (e.g., MCPTool looks for `<tool_call>` tags)

3. **Lazy connection**: MCPTool connects to the server on first use (during `reset()`), not at initialization

4. **Server URL is flexible**: You can specify it in YAML, environment variables, or command-line overrides

5. **Tools add instructions**: The `instruction_string()` method injects tool usage instructions into the agent's prompt

6. **Tool execution is synchronous from env perspective**: Even though MCPTool uses async internally, it appears synchronous to the environment

---

## Example: Creating an MCP-based Environment

You can use **any** existing environment (like `roll_qa`, `roll_math`, `roll_code`) with MCPTool by just adding the `tool_wrapper` config. You don't need to create a special "MCP environment" - you just wrap any environment with MCPTool!

Example:
```yaml
MyTask_with_MCP:
  env_type: "roll_qa"  # Any existing environment
  tool_wrapper:
    wrapper_args:
      tool_reward: 0.0
      max_tool_uses: 3
    tool_configs:
      - tool_id: mcp
        tool_args:
          server_url: ${oc.env:MCP_SERVER_URL}
```

The environment doesn't know about MCP - the `ToolEnvWrapper` handles everything!

