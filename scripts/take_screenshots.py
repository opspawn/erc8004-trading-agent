#!/usr/bin/env python3
"""Take screenshots of the demo server for video creation."""
import os
import sys
from playwright.sync_api import sync_playwright
import time

screenshots_dir = '/home/agent/projects/erc8004-trading-agent/docs/demo-screenshots'
os.makedirs(screenshots_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1920, 'height': 1080})
    page = context.new_page()

    # 1. Judge dashboard - top
    print("Navigating to judge dashboard...")
    page.goto('http://localhost:8084/demo/judge')
    page.wait_for_timeout(4000)
    page.screenshot(path=f'{screenshots_dir}/s54-01-judge-dashboard.png', full_page=False)
    print("Screenshot 1: judge-dashboard")

    # 2. Judge dashboard - scrolled
    page.evaluate('window.scrollBy(0, 500)')
    page.wait_for_timeout(1000)
    page.screenshot(path=f'{screenshots_dir}/s54-02-judge-lower.png', full_page=False)
    print("Screenshot 2: judge-lower")

    # 3. Judge dashboard - more scroll
    page.evaluate('window.scrollBy(0, 500)')
    page.wait_for_timeout(1000)
    page.screenshot(path=f'{screenshots_dir}/s54-03-judge-bottom.png', full_page=False)
    print("Screenshot 3: judge-bottom")

    # 4. Demo UI
    print("Navigating to demo UI...")
    page.goto('http://localhost:8084/demo/ui')
    page.wait_for_timeout(3000)
    page.screenshot(path=f'{screenshots_dir}/s54-04-demo-ui.png', full_page=False)
    print("Screenshot 4: demo-ui")

    # 5. Demo UI scrolled
    page.evaluate('window.scrollBy(0, 500)')
    page.wait_for_timeout(1000)
    page.screenshot(path=f'{screenshots_dir}/s54-05-demo-ui-lower.png', full_page=False)
    print("Screenshot 5: demo-ui-lower")

    # 6. Demo UI - more scroll
    page.evaluate('window.scrollBy(0, 500)')
    page.wait_for_timeout(1000)
    page.screenshot(path=f'{screenshots_dir}/s54-06-demo-ui-signals.png', full_page=False)
    print("Screenshot 6: demo-ui-signals")

    browser.close()

print("All screenshots taken!")
files = sorted([f for f in os.listdir(screenshots_dir) if f.startswith('s54-')])
print(f"Screenshots: {files}")
