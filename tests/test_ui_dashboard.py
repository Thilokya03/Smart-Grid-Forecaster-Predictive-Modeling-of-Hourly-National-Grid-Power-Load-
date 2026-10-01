import time
import pytest
from threading import Thread
from http.server import ThreadingHTTPServer

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from ui.pipeline_dashboard import DashboardHandler, HOST, PORT

SERVER_URL = f"http://{HOST}:{PORT}"

@pytest.fixture(scope="module")
def browser_and_server():
    # Start server in a background thread
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    # Wait for server to bind
    time.sleep(1)

    # Setup headless chrome
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    yield driver
    
    # Teardown
    driver.quit()
    server.shutdown()
    server.server_close()
    server_thread.join()

def test_dashboard_initial_load(browser_and_server):
    driver = browser_and_server
    driver.get(SERVER_URL)
    
    # Wait until the main title is present
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "header h2"))
    )
    
    # Verify title
    assert "UK Smart Grid Forecaster" in driver.title or "Dataset & Forecast Testing UI" in driver.page_source

def test_dashboard_kpis_render(browser_and_server):
    driver = browser_and_server
    driver.get(SERVER_URL)
    
    # Wait for KPIs to render
    kpis_container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "kpis"))
    )
    
    # After JS executes, kpis container should have child cards
    WebDriverWait(driver, 30).until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, "#kpis .card")) > 0
    )
    
    cards = driver.find_elements(By.CSS_SELECTOR, "#kpis .card")
    assert len(cards) > 0

def test_dashboard_time_period_filter(browser_and_server):
    driver = browser_and_server
    driver.get(SERVER_URL)
    
    # Click "Last Month" filter
    last_month_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-period='last_month']"))
    )
    last_month_btn.click()
    
    # Verify active class is applied
    assert "active" in last_month_btn.get_attribute("class")
    
    # Give time for the JS fetch and DOM update
    time.sleep(1)

def test_dashboard_model_toggles(browser_and_server):
    driver = browser_and_server
    driver.get(SERVER_URL)
    
    xgboost_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-model='xgboost']"))
    )
    xgboost_btn.click()
    
    # Verify active class
    assert "model-active" in xgboost_btn.get_attribute("class")
