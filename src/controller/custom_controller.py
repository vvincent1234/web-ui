# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: custom_action.py

import logging
import pyperclip
from main_content_extractor import MainContentExtractor
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.controller.views import ExtractPageContentAction

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self):
        super().__init__()
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard", requires_browser=True)
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        @self.registry.action(
            'Extract page content to get the text or markdown ',
            param_model=ExtractPageContentAction,
            requires_browser=True,
        )
        async def extract_content(params: ExtractPageContentAction, browser: BrowserContext):
            page = await browser.get_current_page()

            content = MainContentExtractor.extract(  # type: ignore
                html=await page.content(),
                output_format=params.value,
            )
            msg = f'📄  Extracted page content\n: {content}\n'
            logger.info(msg)
            # set the extracted content to the memory
            return ActionResult(extracted_content=msg, include_in_memory=True)
