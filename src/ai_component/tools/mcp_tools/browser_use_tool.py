import asyncio
import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from src.ai_component.config import Config
from src.ai_component.exception import CustomException
from src.ai_component.logger import logging

mcp = FastMCP("BrowserUser")

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def start_browser(self, headless: bool = True):
        """Start the browser session"""
        try:
            if not self.playwright:
                self.playwright = await async_playwright().start()
            
            if not self.browser:
                self.browser = await self.playwright.chromium.launch(
                    headless=headless,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
            
            if not self.context:
                self.context = await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
            
            if not self.page:
                self.page = await self.context.new_page()
            
            logging.info("Browser started successfully")
            return True
        except Exception as e:
            logging.error(f"Error starting browser: {e}")
            return False
    
    async def close_browser(self):
        """Close the browser session"""
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
            logging.info("Browser closed successfully")
        except Exception as e:
            logging.error(f"Error closing browser: {e}")

# Global browser manager instance
browser_manager = BrowserManager()

@mcp.tool()
async def navigate_to_url(url: str) -> str:
    """Navigate to a specific URL"""
    try:
        await browser_manager.start_browser()
        
        if not browser_manager.page:
            return "Error: Browser page not available"
        
        logging.info(f'Navigating to URL: {url}')
        await browser_manager.page.goto(url, wait_until='networkidle')
        
        title = await browser_manager.page.title()
        current_url = browser_manager.page.url
        
        return f"Successfully navigated to: {current_url}\nPage title: {title}"
        
    except Exception as e:
        logging.error(f"Error navigating to URL: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def get_page_content(selector: str = "body") -> str:
    """Get text content from the current page or specific element"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info(f'Getting page content with selector: {selector}')
        
        # Wait for the selector to be present
        await browser_manager.page.wait_for_selector(selector, timeout=10000)
        
        # Get text content
        content = await browser_manager.page.text_content(selector)
        
        if content:
            max_length = 5000
            if len(content) > max_length:
                content = content[:max_length] + "... (content truncated)"
            return f"Page content:\n{content.strip()}"
        else:
            return "No content found with the specified selector"
            
    except Exception as e:
        logging.error(f"Error getting page content: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def click_element(selector: str) -> str:
    """Click on an element specified by CSS selector"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info(f'Clicking element with selector: {selector}')
        
        # Wait for element and click
        await browser_manager.page.wait_for_selector(selector, timeout=10000)
        await browser_manager.page.click(selector)
        
        # Wait a moment for any page changes
        await browser_manager.page.wait_for_timeout(1000)
        
        current_url = browser_manager.page.url
        return f"Successfully clicked element. Current URL: {current_url}"
        
    except Exception as e:
        logging.error(f"Error clicking element: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def fill_input(selector: str, text: str) -> str:
    """Fill an input field with specified text"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info(f'Filling input with selector: {selector}')
        
        # Wait for element and fill
        await browser_manager.page.wait_for_selector(selector, timeout=10000)
        await browser_manager.page.fill(selector, text)
        
        return f"Successfully filled input field with: {text}"
        
    except Exception as e:
        logging.error(f"Error filling input: {e}")
        raise CustomException(e, sys) from e


@mcp.tool()
async def wait_for_element(selector: str, timeout: int = 10000) -> str:
    """Wait for an element to appear on the page"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info(f'Waiting for element with selector: {selector}')
        
        await browser_manager.page.wait_for_selector(selector, timeout=timeout)
        
        return f"Element found: {selector}"
        
    except Exception as e:
        logging.error(f"Error waiting for element: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def execute_javascript(script: str) -> str:
    """Execute JavaScript code on the current page"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info('Executing JavaScript code')
        
        result = await browser_manager.page.evaluate(script)
        
        return f"JavaScript executed successfully. Result: {result}"
        
    except Exception as e:
        logging.error(f"Error executing JavaScript: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def get_page_info() -> str:
    """Get basic information about the current page"""
    try:
        if not browser_manager.page:
            return "Error: No active browser session. Navigate to a URL first."
        
        logging.info('Getting page information')
        
        title = await browser_manager.page.title()
        url = browser_manager.page.url
        
        # Get basic page metrics
        viewport = await browser_manager.page.evaluate('''() => {
            return {
                width: window.innerWidth,
                height: window.innerHeight
            }
        }''')
        
        return f"""Page Information:
Title: {title}
URL: {url}
Viewport: {viewport['width']}x{viewport['height']}"""
        
    except Exception as e:
        logging.error(f"Error getting page info: {e}")
        raise CustomException(e, sys) from e

@mcp.tool()
async def close_browser_session() -> str:
    """Close the current browser session"""
    try:
        logging.info('Closing browser session')
        await browser_manager.close_browser()
        return "Browser session closed successfully"
        
    except Exception as e:
        logging.error(f"Error closing browser session: {e}")
        raise CustomException(e, sys) from e

# Cleanup function to ensure browser is closed on exit
async def cleanup():
    """Cleanup function to close browser on exit"""
    await browser_manager.close_browser()

if __name__ == "__main__":
    logging.info("Browser User MCP Tool Started")
    
    try:
        mcp.run(transport="stdio")
    finally:
        asyncio.run(cleanup())