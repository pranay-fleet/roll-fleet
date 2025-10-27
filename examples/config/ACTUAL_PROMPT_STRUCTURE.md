# Actual Prompt Structure for Click and Read Environment

This document shows the **exact** message structure that the LLM receives during training.

## Message Structure Overview

The VLTrajEnvManager constructs messages in this format:
```python
messages = [
    {"role": "system", "content": agent_system_template},
    {"role": "user", "content": [text, image, text]},  # Step 0
    {"role": "assistant", "content": llm_response_0},
    {"role": "user", "content": [text, image, text]},  # Step 1
    {"role": "assistant", "content": llm_response_1},
    ...
]
```

## Episode Start - System Message (Once per episode)

```
ROLE: system

CONTENT:
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
```

## Turn 1 - User Message (Step 0)

**Note**: This is the ONLY turn where `env_instruction` (with task.prompt) is shown!

```
ROLE: user

CONTENT: [
  {
    "type": "text",
    "text": "TASK: Create a new Jira ticket in project CORE for bug XYZ regarding login timeout

You are viewing a web browser screenshot. Analyze the page and take appropriate actions to complete the task.

Available Tools:
1. browser(action=\"left_click\", x=X, y=Y) - Click on elements
2. browser(action=\"right_click\", x=X, y=Y) - Right-click on elements  
3. browser(action=\"double_click\", x=X, y=Y) - Double-click on elements
4. browser(action=\"type\", text=\"...\") - Type text into focused elements
5. browser(action=\"key\", text=\"Enter\") - Press keyboard keys
6. browser(action=\"scroll\", x=X, y=Y, scroll_direction=\"down\", scroll_amount=3) - Scroll pages
7. browser(action=\"wait\", duration=2) - Wait for page updates
8. browser(action=\"left_click_drag\", start_x=X1, start_y=Y1, x=X2, y=Y2) - Drag elements
9. complete_task(success=true/false, summary=\"...\", answer=\"...\") - When done
10. give_up(reason=\"...\", attempts_made=[...]) - Last resort

Examples:
- Click a button: browser(action=\"left_click\", x=150, y=200)
- Right-click menu: browser(action=\"right_click\", x=150, y=200)
- Type in field: browser(action=\"type\", text=\"username\")
- Submit form: browser(action=\"key\", text=\"Enter\")
- Scroll down: browser(action=\"scroll\", x=500, y=400, scroll_direction=\"down\", scroll_amount=3)
- Drag item: browser(action=\"left_click_drag\", start_x=100, start_y=100, x=300, y=300)
- Complete: complete_task(success=true, summary=\"Submitted form\", answer=\"OEP-123\")

--- Turn 1 ---
Screenshot:
"
  },
  {
    "type": "image",
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." # Base64 screenshot
  },
  {
    "type": "text", 
    "text": "
Actions remaining: 10
Max response length: 128 tokens

Decide your next action (use one tool):
"
  }
]
```

## Turn 2 - Assistant Message (Step 0 response)

```
ROLE: assistant

CONTENT:
browser(action="left_click", x=250, y=150)
```

## Turn 3 - User Message (Step 1)

**Note**: No `env_instruction` here - task is NOT repeated!

```
ROLE: user

CONTENT: [
  {
    "type": "text",
    "text": "
--- Turn 2 ---
Screenshot:
"
  },
  {
    "type": "image",
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..." # New screenshot after click
  },
  {
    "type": "text",
    "text": "
Actions remaining: 9
Max response length: 128 tokens

Decide your next action (use one tool):
"
  }
]
```

## Turn 4 - Assistant Message (Step 1 response)

```
ROLE: assistant

CONTENT:
browser(action="type", text="Bug: Login timeout issue")
```

## Turn 5 - User Message (Step 2)

```
ROLE: user

CONTENT: [
  {
    "type": "text",
    "text": "
--- Turn 3 ---
Screenshot:
"
  },
  {
    "type": "image",
    "image": "data:image/jpeg;base64,..." # Screenshot showing typed text
  },
  {
    "type": "text",
    "text": "
Actions remaining: 8
Max response length: 128 tokens

Decide your next action (use one tool):
"
  }
]
```

## Turn 6 - Assistant Message (Step 2 response)

```
ROLE: assistant

CONTENT:
browser(action="key", text="Tab")
```

## ... (Pattern continues) ...

## Final Turn - Assistant Completes Task

```
ROLE: assistant

CONTENT:
complete_task(success=true, summary="Created Jira ticket CORE-1234 for login timeout bug", answer="CORE-1234")
```

---

## Key Points About the Structure

### 1. **env_instruction (task.prompt) - ONLY at Turn 1**
- The dynamic task from Fleet appears ONLY in the first user message
- It's prepended to `pre_step_template` at step 0
- Never repeated in subsequent turns

### 2. **Conversation Flow**
```
System (once)
  ↓
User (Turn 1) ← env_instruction + pre_step + image + next_step
  ↓
Assistant (Turn 1)
  ↓
User (Turn 2) ← pre_step + image + next_step (NO env_instruction)
  ↓
Assistant (Turn 2)
  ↓
... continues until complete_task or max_steps
```

### 3. **Multi-Modal Structure**
Each user message contains 3 parts:
1. **Text before image**: Turn marker (+ env_instruction on turn 1)
2. **Image**: Base64-encoded screenshot
3. **Text after image**: Action counter + prompt for next action

### 4. **What Changes Each Turn**
- Turn index: `--- Turn 1 ---`, `--- Turn 2 ---`, etc.
- Actions remaining: `10`, `9`, `8`, etc.
- Screenshot: New browser state after each action

### 5. **What Stays Constant**
- System message (shown once)
- Structure: text → image → text
- Prompt: "Decide your next action (use one tool):"
- Max response length: 128 tokens

---

## Template Variable Substitution

### pre_step_template (Turn N)
```python
# Config: "\n--- Turn {turn_idx} ---\nScreenshot:\n"

# At step 0 (turn 1):
history.history[0]["env_instruction"] + "\n--- Turn 1 ---\nScreenshot:\n"

# At step 1+ (turn 2+):
"\n--- Turn 2 ---\nScreenshot:\n"
```

### next_step_template
```python
# Config: |
#   Actions remaining: {actions_left}
#   Max response length: {max_response_length} tokens
#   Decide your next action (use one tool):

# Rendered:
f"""
Actions remaining: {10 - step_number}
Max response length: 128 tokens

Decide your next action (use one tool):
"""
```

---

## Example: Complete 3-Turn Episode

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM:
You are an AI assistant that can interact with web browsers...
[Full tool list]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 1):
TASK: Click the Create button
[env_instruction with all tool descriptions]

--- Turn 1 ---
Screenshot:
[IMAGE: Jira home page]
Actions remaining: 10
Decide your next action:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
browser(action="left_click", x=250, y=100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 2):
--- Turn 2 ---
Screenshot:
[IMAGE: Create ticket form opened]
Actions remaining: 9
Decide your next action:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
browser(action="type", text="Bug XYZ")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 3):
--- Turn 3 ---
Screenshot:
[IMAGE: Text entered in form]
Actions remaining: 8
Decide your next action:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
complete_task(success=true, summary="Created ticket", answer="CORE-1234")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## How to Debug the Actual Prompts

To see the exact prompts being generated, you can add logging in `vl_traj_env_manager.py`:

```python
def format_messages(self, history: RolloutCache):
    messages = [...]
    
    # Debug: Print the messages
    import json
    print("\n" + "="*80)
    print("FORMATTED MESSAGES:")
    for msg in messages:
        print(f"\nROLE: {msg['role']}")
        if isinstance(msg['content'], str):
            print(f"CONTENT: {msg['content'][:500]}...")
        else:
            print(f"CONTENT: [Multi-modal with {len(msg['content'])} parts]")
    print("="*80 + "\n")
    
    return lm_input, messages
```

This will show you the exact structure being sent to the tokenizer and model.

