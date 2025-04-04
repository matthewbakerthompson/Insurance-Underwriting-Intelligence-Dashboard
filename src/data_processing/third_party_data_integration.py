"""
Third-Party Data Integration Module

This module provides functionality to integrate various third-party data sources
for enhancing insurance underwriting processes. It includes connectors for:
- Property data APIs
- Weather and catastrophe data
- Business financial data
- Public records data
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ThirdPartyDataConnector:
    """Base class for third-party data connectors."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data connector.
        
        Args:
            api_key: API key for the third-party service
        """
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Some functionality may be limited.")
    
    def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        """
        Make a request to the third-party API.
        
        Args:
            url: The API endpoint URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response data as dictionary
        """
        default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        headers = {**default_headers, **(headers or {})}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}


class PropertyDataConnector(ThirdPartyDataConnector):
    """Connector for property data APIs."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.propertydata.example.com/v1"):
        """
        Initialize the property data connector.
        
        Args:
            api_key: API key for the property data service
            base_url: Base URL for the API
        """
        super().__init__(api_key)
        self.base_url = base_url
    
    def get_property_details(self, address: str) -> Dict:
        """
        Get property details by address.
        
        Args:
            address: Property address
            
        Returns:
            Property details
        """
        endpoint = f"{self.base_url}/properties"
        params = {"address": address}
        
        return self._make_request(endpoint, params)
    
    def get_property_risk_factors(self, property_id: str) -> Dict:
        """
        Get property risk factors.
        
        Args:
            property_id: Property identifier
            
        Returns:
            Property risk factors
        """
        endpoint = f"{self.base_url}/properties/{property_id}/risk_factors"
        
        return self._make_request(endpoint)


class WeatherDataConnector(ThirdPartyDataConnector):
    """Connector for weather and catastrophe data APIs."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.weatherdata.example.com/v2"):
        """
        Initialize the weather data connector.
        
        Args:
            api_key: API key for the weather data service
            base_url: Base URL for the API
        """
        super().__init__(api_key)
        self.base_url = base_url
    
    def get_historical_weather(self, location: str, start_date: str, end_date: str) -> Dict:
        """
        Get historical weather data for a location.
        
        Args:
            location: Location (city, zip code, or coordinates)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Historical weather data
        """
        endpoint = f"{self.base_url}/historical"
        params = {
            "location": location,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return self._make_request(endpoint, params)
    
    def get_catastrophe_risk(self, location: str) -> Dict:
        """
        Get catastrophe risk assessment for a location.
        
        Args:
            location: Location (city, zip code, or coordinates)
            
        Returns:
            Catastrophe risk assessment
        """
        endpoint = f"{self.base_url}/catastrophe_risk"
        params = {"location": location}
        
        return self._make_request(endpoint, params)


class BusinessDataConnector(ThirdPartyDataConnector):
    """Connector for business financial data APIs."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.businessdata.example.com/v1"):
        """
        Initialize the business data connector.
        
        Args:
            api_key: API key for the business data service
            base_url: Base URL for the API
        """
        super().__init__(api_key)
        self.base_url = base_url
    
    def get_business_profile(self, business_id: str) -> Dict:
        """
        Get business profile information.
        
        Args:
            business_id: Business identifier
            
        Returns:
            Business profile data
        """
        endpoint = f"{self.base_url}/businesses/{business_id}"
        
        return self._make_request(endpoint)
    
    def get_financial_health(self, business_id: str) -> Dict:
        """
        Get business financial health indicators.
        
        Args:
            business_id: Business identifier
            
        Returns:
            Financial health indicators
        """
        endpoint = f"{self.base_url}/businesses/{business_id}/financial_health"
        
        return self._make_request(endpoint)


class DataIntegrator:
    """
    Integrates data from multiple third-party sources to create
    comprehensive profiles for underwriting.
    """
    
    def __init__(self):
        """Initialize the data integrator with connectors."""
        self.property_connector = PropertyDataConnector()
        self.weather_connector = WeatherDataConnector()
        self.business_connector = BusinessDataConnector()
    
    def create_property_risk_profile(self, address: str) -> Dict:
        """
        Create a comprehensive property risk profile.
        
        Args:
            address: Property address
            
        Returns:
            Comprehensive property risk profile
        """
        # Get basic property details
        property_details = self.property_connector.get_property_details(address)
        
        if "error" in property_details:
            return property_details
        
        property_id = property_details.get("property_id")
        location = property_details.get("location", {})
        coordinates = location.get("coordinates", {})
        
        # Get property risk factors
        risk_factors = self.property_connector.get_property_risk_factors(property_id)
        
        # Get catastrophe risk
        location_str = f"{coordinates.get('latitude')},{coordinates.get('longitude')}"
        catastrophe_risk = self.weather_connector.get_catastrophe_risk(location_str)
        
        # Combine all data
        return {
            "property_details": property_details,
            "risk_factors": risk_factors,
            "catastrophe_risk": catastrophe_risk
        }
    
    def create_business_risk_profile(self, business_id: str, address: str) -> Dict:
        """
        Create a comprehensive business risk profile.
        
        Args:
            business_id: Business identifier
            address: Business address
            
        Returns:
            Comprehensive business risk profile
        """
        # Get business profile
        business_profile = self.business_connector.get_business_profile(business_id)
        
        if "error" in business_profile:
            return business_profile
        
        # Get financial health
        financial_health = self.business_connector.get_financial_health(business_id)
        
        # Get property risk profile
        property_risk = self.create_property_risk_profile(address)
        
        # Combine all data
        return {
            "business_profile": business_profile,
            "financial_health": financial_health,
            "property_risk": property_risk
        }
    
    def process_and_save_data(self, data: Dict, output_path: str) -> None:
        """
        Process and save integrated data to a file.
        
        Args:
            data: Integrated data
            output_path: Path to save the processed data
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the data
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    integrator = DataIntegrator()
    
    # Create a property risk profile
    property_profile = integrator.create_property_risk_profile("123 Main St, Anytown, USA")
    integrator.process_and_save_data(
        property_profile,
        "../data/processed/property_risk_profile.json"
    )
    
    # Create a business risk profile
    business_profile = integrator.create_business_risk_profile(
        "B12345",
        "456 Business Ave, Commerce City, USA"
    )
    integrator.process_and_save_data(
        business_profile,
        "../data/processed/business_risk_profile.json"
    )
