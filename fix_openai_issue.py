#!/usr/bin/env python3
"""
Fix OpenAI proxy issue
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check for proxy-related environment variables"""
    print("🔍 Checking environment for proxy issues...")
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    found_proxies = []
    
    for var in proxy_vars:
        if var in os.environ:
            found_proxies.append((var, os.environ[var]))
            print(f"  ❌ Found {var}: {os.environ[var]}")
    
    if not found_proxies:
        print("  ✅ No proxy environment variables found")
    
    return found_proxies

def check_openai_config():
    """Check for OpenAI configuration files"""
    print("\n🔍 Checking for OpenAI configuration files...")
    
    config_locations = [
        os.path.expanduser("~/.openai"),
        os.path.expanduser("~/.config/openai"),
        "/etc/openai",
        "/usr/local/etc/openai"
    ]
    
    found_configs = []
    for location in config_locations:
        if os.path.exists(location):
            found_configs.append(location)
            print(f"  ❌ Found config at: {location}")
    
    if not found_configs:
        print("  ✅ No OpenAI config files found")
    
    return found_configs

def test_openai_import():
    """Test OpenAI import and basic functionality"""
    print("\n🔍 Testing OpenAI import...")
    
    try:
        import openai
        print(f"  ✅ OpenAI version: {openai.__version__}")
        
        # Test client creation
        from openai import OpenAI
        print("  ✅ OpenAI client import successful")
        
        return True
    except Exception as e:
        print(f"  ❌ OpenAI import error: {e}")
        return False

def test_openai_client():
    """Test OpenAI client creation"""
    print("\n🔍 Testing OpenAI client creation...")
    
    try:
        from openai import OpenAI
        
        # Clear any proxy environment variables
        env_vars_to_clear = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
        
        # Try creating client
        client = OpenAI(api_key="test-key")
        print("  ✅ OpenAI client creation successful")
        return True
        
    except Exception as e:
        print(f"  ❌ OpenAI client creation error: {e}")
        return False

def fix_openai_issue():
    """Attempt to fix OpenAI issue"""
    print("\n🔧 Attempting to fix OpenAI issue...")
    
    # Clear all proxy environment variables
    env_vars_to_clear = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            print(f"  🗑️  Cleared {var}")
    
    # Set explicit no-proxy
    os.environ['NO_PROXY'] = '*'
    print("  ✅ Set NO_PROXY=*")
    
    return True

def main():
    """Main diagnostic function"""
    print("🔧 OpenAI Issue Diagnostic Tool")
    print("=" * 40)
    
    # Check environment
    proxies = check_environment()
    
    # Check config files
    configs = check_openai_config()
    
    # Test imports
    import_ok = test_openai_import()
    
    # Test client creation
    client_ok = test_openai_client()
    
    # Attempt fix
    if not client_ok:
        fix_openai_issue()
        print("\n🔄 Testing again after fix...")
        client_ok = test_openai_client()
    
    # Summary
    print("\n📊 Summary:")
    if client_ok:
        print("  ✅ OpenAI client should work now")
        print("  💡 Try running the evaluation again")
    else:
        print("  ❌ OpenAI client still has issues")
        print("  💡 Consider reinstalling openai package: pip install --upgrade openai")

if __name__ == "__main__":
    main()
