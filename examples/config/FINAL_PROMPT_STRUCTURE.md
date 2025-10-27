# ✅ CORRECTED Prompt Structure (No Tool Duplication)

## The Problem We Fixed
Previously, tools were listed in BOTH places:
- ❌ System prompt (agent_system_template) 
- ❌ env_instruction (get_instructions())

This was redundant and wasted tokens!

## Current Clean Structure

### 1. System Message (Once, tools defined here)
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
   - complete_task(success=true, summary="Created Jira ticket", answer="OEP-130")

3. give_up - LAST RESORT: Give up on the task when you have absolutely no other alternatives:
   - give_up(reason="detailed explanation", attempts_made=["list", "of", "attempts"])
   - Only use when: 1) All approaches exhausted, 2) Task appears impossible, 3) Stuck in unrecoverable state

Important guidelines:
- Analyze the screenshot carefully before taking actions
- Be methodical and break down tasks into steps
- Use coordinates within screen bounds
- Call complete_task when the task is done
```

### 2. User Message - Turn 1 (Step 0) - Task only, tools NOT repeated

```
ROLE: user

CONTENT: [
  {
    "type": "text",
    "text": "TASK: Create a new Jira ticket in project CORE for bug XYZ regarding login timeout

Analyze the screenshot carefully and use the available tools to complete this task step by step.

--- Turn 1 ---
Screenshot:
"
  },
  {
    "type": "image", 
    "image": "data:image/jpeg;base64,..."
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

### 3. Assistant Message - Turn 1

```
ROLE: assistant

CONTENT:
browser(action="left_click", x=250, y=150)
```

### 4. User Message - Turn 2+ (Step 1+) - No task, no tools

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
    "image": "data:image/jpeg;base64,..."
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

## Summary of Where Things Appear

| Content | System | Turn 1 | Turn 2+ |
|---------|--------|--------|---------|
| Tool descriptions (10 tools) | ✅ Once | ❌ No | ❌ No |
| Task prompt (dynamic from Fleet) | ❌ No | ✅ Once | ❌ No |
| Turn marker | ❌ No | ✅ Yes | ✅ Yes |
| Screenshot | ❌ No | ✅ Yes | ✅ Yes |
| Actions remaining | ❌ No | ✅ Yes | ✅ Yes |
| Action prompt | ❌ No | ✅ Yes | ✅ Yes |

## Token Efficiency

**Before (with duplication):**
- System: ~300 tokens (tools)
- Turn 1: ~350 tokens (task + tools + turn info)
- Turn 2+: ~50 tokens (turn info only)

**After (no duplication):**
- System: ~300 tokens (tools)
- Turn 1: ~100 tokens (task + turn info)
- Turn 2+: ~50 tokens (turn info only)

**Savings**: ~250 tokens per episode at Turn 1! 🎉

## Complete Example Episode

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM (Once):
You are an AI assistant that can interact with web browsers...

You have access to the following tools:
1. browser(action="left_click"...)
2. browser(action="right_click"...)
[...all 10 tools listed with examples...]

Important guidelines: ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 1):
TASK: Create a Jira ticket for bug XYZ

Analyze the screenshot carefully and use the available tools 
to complete this task step by step.

--- Turn 1 ---
Screenshot:
[IMAGE: Jira homepage]

Actions remaining: 10
Decide your next action (use one tool):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
browser(action="left_click", x=250, y=100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 2):
--- Turn 2 ---
Screenshot:
[IMAGE: Create form opened]

Actions remaining: 9
Decide your next action (use one tool):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
browser(action="type", text="Bug XYZ")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER (Turn 3):
--- Turn 3 ---
Screenshot:
[IMAGE: Text entered]

Actions remaining: 8
Decide your next action (use one tool):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSISTANT:
complete_task(success=true, summary="Created ticket", answer="CORE-1234")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Key Points

✅ **Tools appear once** in system prompt  
✅ **Task appears once** at Turn 1  
✅ **No repetition** in subsequent turns  
✅ **Efficient token usage**  
✅ **Clean conversation structure**  

The LLM:
1. Learns tools from system message (persistent context)
2. Gets specific task at Turn 1 
3. Sees updated screenshots each turn
4. Issues tool commands to complete task
5. Calls complete_task when done

Perfect! 🚀

