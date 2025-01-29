import pdb
from typing import List, Optional

from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.views import BrowserState
from langchain_core.messages import HumanMessage, SystemMessage

from .custom_views import CustomAgentStepInfo


class CustomSystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = """
    1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
       {
         "current_state": {
           "prev_action_evaluation": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not. Note that the result you output must be consistent with the reasoning you output afterwards. If you consider it to be 'Failed,' you should reflect on this during your thought.",
           "important_contents": "Output important contents closely related to user\'s instruction on the current page. If there is, please output the contents. If not, please output empty string ''.",
           "task_progress": "Task Progress is a general summary of the current contents that have been completed. Just summarize the contents that have been actually completed based on the content at current step and the history operations. Please list each completed item individually, such as: 1. Input username. 2. Input Password. 3. Click confirm button. Please return string type not a list.",
           "future_plans": "Based on the user's request and the current state, outline the remaining steps needed to complete the task. This should be a concise list of actions yet to be performed, such as: 1. Select a date. 2. Choose a specific time slot. 3. Confirm booking. Please return string type not a list.",
           "thought": "Think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next one operation. If your output of prev_action_evaluation is 'Failed', please reflect and output your reflection here.",
           "summary": "Please generate a brief natural language description for the operation in next actions based on your Thought."
         },
         "action": [
           * actions in sequences, please refer to **Common action sequences**. Each output action MUST be formated as: \{action_name\: action_params\}* 
         ]
       }

    2. ACTIONS: You can specify multiple actions to be executed in sequence. 

       Common action sequences:
       - Form filling: [
           {"input_text": {"index": 1, "text": "username"}},
           {"input_text": {"index": 2, "text": "password"}},
           {"click_element": {"index": 3}}
         ]
       - Navigation and extraction: [
           {"go_to_url": {"url": "https://example.com"}},
           {"extract_page_content": {}}
         ]


    3. ELEMENT INTERACTION:
       - Only use indexes that exist in the provided element list
       - Each element has a unique index number (e.g., "33[:]<button>")
       - Elements marked with "_[:]" are non-interactive (for context only)

    4. NAVIGATION & ERROR HANDLING:
       - If no suitable elements exist, use other functions to complete the task
       - If stuck, try alternative approaches
       - Handle popups/cookies by accepting or closing them
       - Use scroll to find elements you are looking for

    5. TASK COMPLETION:
       - If you think all the requirements of user\'s instruction have been completed and no further operation is required, output the **Done** action to terminate the operation process.
       - Don't hallucinate actions.
       - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
       - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.
       - Note that you must verify if you've truly fulfilled the user's request by examining the actual page content, not just by looking at the actions you output but also whether the action is executed successfully. Pay particular attention when errors occur during action execution.

    6. VISUAL CONTEXT:
       - When an image is provided, use it to understand the page layout
       - Bounding boxes with labels correspond to element indexes
       - Each bounding box and its label have the same color
       - Most often the label is inside the bounding box, on the top right
       - Visual context helps verify element locations and relationships
       - sometimes labels overlap, so use the context to verify the correct element

    7. Form filling:
       - If you fill an input field and your action sequence is interrupted, most often a list with suggestions poped up under the field and you need to first select the right element from the suggestion list.

    8. ACTION SEQUENCING:
       - Actions are executed in the order they appear in the list 
       - Each action should logically follow from the previous one
       - If the page changes after an action, the sequence is interrupted and you get the new state.
       - If content only disappears the sequence continues.
       - Only provide the action sequence until you think the page will change.
       - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
       - only use multiple actions if it makes sense. 
    """
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
    INPUT STRUCTURE:
    1. Task: The user\'s instructions you need to complete.
    2. Hints(Optional): Some hints to help you complete the user\'s instructions.
    3. Memory: Important contents are recorded during historical operations for use in subsequent operations.
    4. Current URL: The webpage you're currently on
    5. Available Tabs: List of open browser tabs
    6. Interactive Elements: List in the format:
       index[:]<element_type>element_text</element_type>
       - index: Numeric identifier for interaction
       - element_type: HTML element type (button, input, etc.)
       - element_text: Visible text or element description

    Example:
    33[:]<button>Submit Form</button>
    _[:] Non-interactive text


    Notes:
    - Only elements with numeric indexes are interactive
    - _[:] elements provide context but cannot be interacted with
    """

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            str: Formatted system prompt
        """
        time_str = self.current_date.strftime("%Y-%m-%d %H:%M")

        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
    1. Analyze the provided webpage elements and structure
    2. Plan a sequence of actions to accomplish the given task
    3. Your final result MUST be a valid JSON as the **RESPONSE FORMAT** described, containing your action sequence and state assessment, No need extra content to expalin. 

    Current date and time: {time_str}

    {self.input_format()}

    {self.important_rules()}

    Functions:
    {self.default_action_description}

    Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
        return SystemMessage(content=AGENT_PROMPT)


class CustomAgentMessagePrompt(AgentMessagePrompt):
    def __init__(
            self,
            state: BrowserState,
            actions: Optional[List[ActionModel]] = None,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            step_info: Optional[CustomAgentStepInfo] = None,
    ):
        super(CustomAgentMessagePrompt, self).__init__(state=state, 
                                                       result=result, 
                                                       include_attributes=include_attributes, 
                                                       max_error_length=max_error_length, 
                                                       step_info=step_info
                                                       )
        self.actions = actions

    def get_user_message(self) -> HumanMessage:
        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number}/{self.step_info.max_steps}\n'
        else:
            step_info_description = ''

        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'
   
        state_description = f"""
{step_info_description}
1. Task: {self.step_info.task}. 
2. Hints(Optional): 
{self.step_info.add_infos}
3. Memory: 
{self.step_info.memory}
4. Current url: {self.state.url}
5. Available tabs:
{self.state.tabs}
6. Interactive elements:
{elements_text}
        """

        if self.actions and self.result:
            state_description += "\n **Previous Actions** \n"
            state_description += f'Previous step: {self.step_info.step_number-1}/{self.step_info.max_steps} \n'
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {result.extracted_content}\n"
                    if result.error:
                        # only use last 300 characters of error
                        error = result.error[-self.max_error_length:]
                        state_description += (
                            f"Error of previous action {i + 1}/{len(self.result)}: ...{error}\n"
                        )

        if self.state.screenshot:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_description},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.state.screenshot}"
                        },
                    },
                ]
            )

        return HumanMessage(content=state_description)
    
class MonitorSystemPrompt(CustomSystemPrompt):
    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            str: Formatted system prompt
        """
        system_prompt = """
    You are a helpful AI assistant tasked with monitoring an agent's execution of a user's instruction. Your role is to provide structured updates on the evaluation of past actions, important contents, the task's progress, future plans, and overall task status. Base your analysis on the current user's input and historical information.

    **Input Structure:**
    1. Task: The user's instructions that need to be completed.
    2. Previous Actions: The last actions and their results taken by the execution agent.
    3. Curent Interactive Elements: A list of interactive elements in the following format:
       index[:]<element_type>element_text</element_type>
        - index: A numeric identifier for the interactive element.
        - element_type: The HTML element type (e.g., button, input, link).
        - element_text: The visible text or description of the element.

        Example:
        33[:]<button>Submit Form</button>
        _[:] Non-interactive text


        Notes:
        - Only elements with numeric indexes are interactive
        - _[:] elements provide context but cannot be interacted with

    **Final Output Requirements:**
    Your final output MUST be a JSON object with the following keys:

    a. 'prev_action_evaluation': An assessment of the last action taken by the agent. Choose from "Success", "Failed", or "Unknown". If the action failed, provide a reason and suggestions for improvement or modification.
    b. 'important_contents': A list of important contents closely related to or useful to user\'s instruction on the current page.
    c. 'task_progress': A list of sub-tasks that have been successfully completed so far. Describe each completed sub-task. Output an empty list if no tasks are completed.
    d. 'future_plans': A list of future steps required to complete the user's task. Describe these steps in natural language.
    e. 'is_done': Yes or No, indicating whether the user's task has been fully completed. Set to Yes only when all task requirements are met. If Yes, return the requested information for the user. If No, provide a brief explanation of why the task is not yet complete.

    **Example Final Output:**
    ```json
    {
        "prev_action_evaluation": "Success - The browser home page have been opened.",
        "important_contents": [],
        "task_progress": ["Open google home page"],
        "future_plans": ["Type OpenAI in search bar", "Click the 'Search' button to search"],
        "is_done": "No - The page is still not available."
    }
    ```

    **Reasoning and Thinking Guidance:**

    *   Begin by carefully analyzing the user's `Task` and overall intent.
    *   Then sequentially analyze, and output the JSON keys: `prev_action_evaluation`, `important_contents`, `task_progress`, `future_plans`, and `is_done`.
    
    Remember: Your Final output must be valid JSON matching the specified format. Each action in the sequence must be valid.
    """
        return SystemMessage(content=system_prompt)
    

class MonitorAgentMessagePrompt(CustomAgentMessagePrompt):
    def get_user_message(self) -> HumanMessage:
        state_description = f'Current step: {self.step_info.step_number}/{self.step_info.max_steps}'
        state_description += f"1. Task: {self.step_info.task} \n"
        if self.actions and self.result:
            state_description += f"2. Previous Actions in Previou Step {self.step_info.step_number-1}: \n"
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {result.extracted_content} \n"
                    if result.error:
                        # only use last 300 characters of error
                        error = result.error[-self.max_error_length:]
                        state_description += (
                            f"Error of previous action {i + 1}/{len(self.result)}: ...{error} \n"
                        )
        else:
            state_description += "2. Previous Actions: No previous actions. \n"

        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)
        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0
        
        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'
            
        state_description += f"3. Interactive elements in Step {self.step_info.step_number}: \n{elements_text}\n"

        if self.state.screenshot:
            # Format message for vision model
            content = [
                {"type": "text", "text": state_description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{state.screenshot}"
                    },
                },
            ]
            return HumanMessage(content=content)
        else:
            return HumanMessage(content=state_description)

    
class CustomSystemPromptV2(CustomSystemPrompt):
    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = """
    1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
       {
         "current_state": {
           "thought": "Think about the requirements that have been completed in previous operations and the requirements that need to be completed in the next operation.",
           "summary": "Please generate a brief natural language description for the operation in next actions based on your thought."
         },
         "action": [
           * actions in sequences, please refer to **Common action sequences**. Each output action MUST be formated as: \{action_name\: action_params\}* 
         ]
       }

    2. ACTIONS: You can specify multiple actions to be executed in sequence. 

       Common action sequences:
       - Form filling: [
           {"input_text": {"index": 1, "text": "username"}},
           {"input_text": {"index": 2, "text": "password"}},
           {"click_element": {"index": 3}}
         ]
       - Navigation and extraction: [
           {"go_to_url": {"url": "https://example.com"}},
           {"extract_page_content": {}}
         ]


    3. ELEMENT INTERACTION:
       - Only use indexes that exist in the provided element list
       - Each element has a unique index number (e.g., "33[:]<button>")
       - Elements marked with "_[:]" are non-interactive (for context only)

    4. NAVIGATION & ERROR HANDLING:
       - If no suitable elements exist, use other functions to complete the task
       - If stuck, try alternative approaches
       - Handle popups/cookies by accepting or closing them
       - Use scroll to find elements you are looking for

    5. TASK COMPLETION:
       - If you think all the requirements of user\'s instruction have been completed and no further operation is required, output the done action to terminate the operation process.
       - Don't hallucinate actions.
       - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
       - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.

    6. VISUAL CONTEXT:
       - When an image is provided, use it to understand the page layout
       - Bounding boxes with labels correspond to element indexes
       - Each bounding box and its label have the same color
       - Most often the label is inside the bounding box, on the top right
       - Visual context helps verify element locations and relationships
       - sometimes labels overlap, so use the context to verify the correct element

    7. Form filling:
       - If you fill an input field and your action sequence is interrupted, most often a list with suggestions poped up under the field and you need to first select the right element from the suggestion list.

    8. ACTION SEQUENCING:
       - Actions are executed in the order they appear in the list 
       - Each action should logically follow from the previous one
       - If the page changes after an action, the sequence is interrupted and you get the new state.
       - If content only disappears the sequence continues.
       - Only provide the action sequence until you think the page will change.
       - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
       - only use multiple actions if it makes sense. 
    """
        text += f"   - use maximum {self.max_actions_per_step} actions per sequence"
        return text

    def input_format(self) -> str:
        return """
    INPUT STRUCTURE:
    1. Task: The user\'s instructions you need to complete.
    2. Hints(Optional): Some hints to help you complete the user\'s instructions.
    3. Previous Action Evaluation: The evaluation of the last action.
    4. Memory: Important contents are recorded during historical operations for use in subsequent operations.
    5. Task Progress: Up to the current page, the content you have completed can be understood as the progress of the task.
    6. Future Plans: The next steps you need to take to complete the task.
    7. Current URL: The webpage you're currently on
    8. Available Tabs: List of open browser tabs
    9. Interactive Elements: List in the format:
       index[:]<element_type>element_text</element_type>
       - index: Numeric identifier for interaction
       - element_type: HTML element type (button, input, etc.)
       - element_text: Visible text or element description

    Example:
    33[:]<button>Submit Form</button>
    _[:] Non-interactive text


    Notes:
    - Only elements with numeric indexes are interactive
    - _[:] elements provide context but cannot be interacted with
    """

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            str: Formatted system prompt
        """
        time_str = self.current_date.strftime("%Y-%m-%d %H:%M")

        AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
    1. Analyze the provided webpage elements and structure
    2. Plan a sequence of actions to accomplish the given task
    3. Respond with valid JSON containing your action sequence and state assessment

    Current date and time: {time_str}

    {self.input_format()}

    {self.important_rules()}

    Functions:
    {self.default_action_description}

    Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
        return SystemMessage(content=AGENT_PROMPT)


class CustomAgentMessagePromptV2(CustomAgentMessagePrompt):
    def get_user_message(self) -> HumanMessage:
        state_description = f"Current Step: {self.step_info.step_number}/{self.step_info.max_steps}\n"
        elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

        has_content_above = (self.state.pixels_above or 0) > 0
        has_content_below = (self.state.pixels_below or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Start of page]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[End of page]'
        else:
            elements_text = 'empty page'
        state_description += f"""
    1. Task: {self.step_info.task}
    2. Hints(Optional): 
    {self.step_info.add_infos}
    3. Previous Action Evaluation:
    {self.step_info.prev_action_evaluation}
    4. Memory: 
    {self.step_info.memory}
    5. Task Progress: 
    {self.step_info.task_progress}
    6. Future Plans:
    {self.step_info.future_plans}
    7. Current url: {self.state.url}
    8. Available tabs:
    {self.state.tabs}
    9. Interactive elements:
    {elements_text}
            """

        if self.actions and self.result:
            state_description += "\n **Previous Actions** \n"
            state_description += f'Previous step: {self.step_info.step_number-1}/{self.step_info.max_steps}'
            for i, result in enumerate(self.result):
                action = self.actions[i]
                state_description += f"Previous action {i + 1}/{len(self.result)}: {action.model_dump_json(exclude_unset=True)}\n"
                if result.include_in_memory:
                    if result.extracted_content:
                        state_description += f"Result of previous action {i + 1}/{len(self.result)}: {result.extracted_content}\n"
                    if result.error:
                        # only use last 300 characters of error
                        error = result.error[-self.max_error_length:]
                        state_description += (
                            f"Error of previous action {i + 1}/{len(self.result)}: ...{error}\n"
                        )

        if self.state.screenshot:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {"type": "text", "text": state_description},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.state.screenshot}"
                        },
                    },
                ]
            )

        return HumanMessage(content=state_description)