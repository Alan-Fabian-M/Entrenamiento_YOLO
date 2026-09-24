#!/usr/bin/env python3
"""Test script for Backend API."""

import argparse
import json
import sys
from pathlib import Path

import requests


DEFAULT_URL = "http://localhost:8000"


def test_health(url: str = DEFAULT_URL) -> bool:
    """Test health endpoint."""
    try:
        response = requests.get(f"{url}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Project: {data.get('project')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {url}")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_compilar_sala(image_path: str, url: str = DEFAULT_URL) -> bool:
    """Test compilar-sala endpoint."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        print(f"Processing image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{url}/api/v1/compilar-sala",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Scene compilation successful")
            print(f"   Room dimensions: {data['room_info']['dimensions']}")
            print(f"   Furniture detected: {data['metadata']['furniture_count']}")
            print(f"   Processing time: {data['metadata']['processing_time_ms']:.1f}ms")
            print(f"   Scale confidence: {data['metadata']['scale_confidence']:.2f}")
            return True
        else:
            print(f"❌ Scene compilation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Backend API")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--image", help="Path to floor plan image")
    args = parser.parse_args()
    
    print(f"Testing API at: {args.url}\n")
    health_ok = test_health(args.url)
    
    if not health_ok:
        return 1
    
    print()
    if args.image:
        test_compilar_sala(args.image, args.url)
    else:
        print("Tip: python3 test_api.py --image imagenes/tu_imagen.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
