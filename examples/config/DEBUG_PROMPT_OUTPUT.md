# Debug Output: Prompt Structure at Each Step

## What Was Added

Debug print statements in `vl_traj_env_manager.py` to show exactly what the LLM sees at each step.

## Output Format

When you run training, you'll see output like this:

```
================================================================================
🔷 SYSTEM MESSAGE (Step 0 - Shown Once)
================================================================================
You are an AI assistant that can interact with web browsers to solve tasks.

You have access to the following tools:

1. browser - Interact with the web page:
   - browser(action="left_click", x=X, y=Y) - Click at coordinates
   - browser(action="right_click", x=X, y=Y) - Right-click on elements
   - browser(action="double_click", x=X, y=Y) - Double-click on elements
   - browser(action="type", text="your text") - Type text into focused elements
   - browser(action="key", text="Enter") - Press keys (Enter, Escape, Tab, etc.)
   - browser(action="scroll", x=X, y=Y, scroll_direction="down", scroll_amount=3) - Scroll pages
   - browser(action="wait", duration=2) - Wait for page updates
   - browser(action="left_click_drag", start_x=X1, start_y=Y1, x=X2, y=Y2) - Drag elements

2. complete_task - Signal when you have successfully completed the task:
   - complete_task(success=true/false, summary="description of what was accomplished")
   - complete_task(success=true, summary="Created Jira ticket", answer="OEP-130") - Include answer if task asks for specific data

3. give_up - LAST RESORT: Give up on the task when you have absolutely no other alternatives:
   - give_up(reason="detailed explanation", attempts_made=["list", "of", "specific attempts"])
   - Only use when: 1) All approaches exhausted, 2) Task appears impossible, 3) Stuck in unrecoverable state

Important guidelines:
- Analyze the screenshot carefully before taking actions
- Be methodical and break down tasks into steps
- Use coordinates within screen bounds
- Call complete_task when the task is done
================================================================================


================================================================================
👤 USER MESSAGE (Step 0 - Turn 1)
================================================================================
Pre-step text:
TASK: Create a new Jira ticket in project CORE for bug XYZ regarding login timeout

Analyze the screenshot carefully and use the available tools to complete this task step by step.

--- Turn 1 ---
Screenshot:

[IMAGE: Screenshot (850, 1400, 3)]

Next-step text:

Actions remaining: 10
Max response length: 128 tokens

Decide your next action (use one tool):

================================================================================


================================================================================
🤖 ASSISTANT RESPONSE (Step 0)
================================================================================
browser(action="left_click", x=250, y=150)
================================================================================


================================================================================
👤 USER MESSAGE (Step 1 - Turn 2)
================================================================================
Pre-step text:

--- Turn 2 ---
Screenshot:

[IMAGE: Screenshot (850, 1400, 3)]

Next-step text:

Actions remaining: 9
Max response length: 128 tokens

Decide your next action (use one tool):

================================================================================


================================================================================
🤖 ASSISTANT RESPONSE (Step 1)
================================================================================
browser(action="type", text="Bug: Login timeout issue")
================================================================================


================================================================================
👤 USER MESSAGE (Step 2 - Turn 3)
================================================================================
Pre-step text:

--- Turn 3 ---
Screenshot:

[IMAGE: Screenshot (850, 1400, 3)]

Next-step text:

Actions remaining: 8
Max response length: 128 tokens

Decide your next action (use one tool):

================================================================================


================================================================================
🤖 ASSISTANT RESPONSE (Step 2)
================================================================================
complete_task(success=true, summary="Created Jira ticket CORE-1234", answer="CORE-1234")
================================================================================
```

## What You Can Observe

### Step 0 (Turn 1) - Episode Start
✅ **System message** printed with full tool list  
✅ **User message** includes:
   - `env_instruction` with TASK + brief instruction
   - Turn marker: "--- Turn 1 ---"
   - Screenshot placeholder
   - Action counter: "Actions remaining: 10"
✅ **Assistant response** with first action

### Step 1+ (Turn 2+) - Subsequent Turns
✅ **User message** includes:
   - NO task (not repeated!)
   - Turn marker: "--- Turn 2 ---", "--- Turn 3 ---", etc.
   - Screenshot placeholder
   - Action counter: "Actions remaining: 9", "8", etc.
✅ **Assistant response** with next action

### Key Observations:

1. **System message**: Shown ONCE at step 0
2. **Task (env_instruction)**: Shown ONCE at step 0 in first user message
3. **Tools**: Shown ONCE in system message (NOT repeated in user messages)
4. **Turn markers**: Shown at EVERY step
5. **Screenshots**: New one at EVERY step
6. **Action counter**: Decrements at EVERY step

## Verifying the Structure

When you run training, look for these patterns:

### ✅ Expected at Step 0:
- System message with tools (once)
- User message with "TASK: ..." 
- No tool repetition in user message (clean!)

### ✅ Expected at Step 1+:
- No system message (already shown)
- User message with just turn info
- No task, no tools

### ❌ Should NOT see:
- Tools listed in user messages
- Task repeated after step 0
- System message repeated

## Debugging Tips

If you see unexpected behavior:

1. **Task appearing multiple times?**
   - Check that `env_instruction` is only prepended when `step == 0`
   - Line 162-163 in vl_traj_env_manager.py

2. **Tools in user messages?**
   - Check `get_instructions()` in env.py
   - Should only return task, not tool list

3. **Too many tokens?**
   - Count tokens in system message (~300)
   - Count tokens in step 0 user message (~100)
   - Count tokens in step 1+ user messages (~50)

## Disable Debug Prints

Once you've verified the structure, you can comment out the print statements:

```python
# In vl_traj_env_manager.py, comment out lines 148-154, 184-191, 196-201
```

Or wrap them in a debug flag:

```python
DEBUG_PROMPTS = False

if DEBUG_PROMPTS and self.rollout_cache.step == 0:
    print(...)
```

## Token Efficiency Check

Based on the debug output, verify:
- System: ~300 tokens (tools + guidelines)
- Step 0: ~100 tokens (task + turn info)
- Step 1+: ~50 tokens (turn info only)

Total for 3-step episode: ~300 + 100 + 50 + 50 = ~500 tokens of text (plus image tokens)

Much better than if we repeated tools at each step! 🎉

