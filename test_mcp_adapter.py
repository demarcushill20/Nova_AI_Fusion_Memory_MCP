#!/usr/bin/env python3
"""
MCP Tool Test Script
--------------------
This script tests MCP tool calls by sending MCP-formatted requests
to the server process via stdin and reading responses from stdout.

Usage:
  - Start the MCP server in a separate terminal using:
    `docker-compose --profile mcp up -d`
  - Run this script:
    `python test_mcp_adapter.py`

This simulates how Claude Desktop would interact with the MCP server.
"""

import json
import subprocess
import uuid
import time


def run_mcp_command(command, args=None):
    """
    Run an MCP command by sending it to the server via stdin
    and reading the response from stdout.
    
    Args:
        command: The MCP command/tool to run
        args: The arguments for the command (as a dictionary)
        
    Returns:
        The parsed JSON response from the MCP server
    """
    if args is None:
        args = {}
    
    # Create the MCP request
    request = {
        "tool": command,
        "args": args
    }
    
    # Convert the request to JSON
    request_json = json.dumps(request)
    
    # Use subprocess to run the docker command
    # This is similar to how Claude Desktop would run the command
    cmd = [
        "docker", "run", "-i", "--rm", "--network=host",
        "nova-memory-mcp:latest"
    ]
    
    try:
        # Start the process
        print(f"Starting Docker process with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Write the request to stdin
        print(f"Sending request: {request_json}")
        process.stdin.write(request_json + "\n")
        process.stdin.flush()
        
        # Wait for the process to initialize (give the server time to start)
        print("Waiting for MCP server to initialize...")
        time.sleep(3)
        
        # Read the response from stdout with timeout
        print("Reading response...")
        
        # Read any stderr output for debugging
        stderr_output = ""
        while process.poll() is None and process.stderr.readable():
            line = process.stderr.readline()
            if line:
                stderr_output += line
            else:
                break
                
        if stderr_output:
            print(f"STDERR output:\n{stderr_output}")
            
        # Read stdout
        stdout_output = ""
        timeout = 10  # 10 seconds timeout
        start_time = time.time()
        
        while process.poll() is None and (time.time() - start_time) < timeout:
            if process.stdout.readable():
                line = process.stdout.readline()
                if line:
                    stdout_output += line
                    break
            time.sleep(0.1)
            
        print(f"Raw response: {stdout_output}")
        
        # Parse the response as JSON if we got something
        if stdout_output.strip():
            try:
                response = json.loads(stdout_output)
            except json.JSONDecodeError as e:
                response = {"error": f"Failed to parse JSON response: {e}", "raw_output": stdout_output}
        else:
            response = {"error": "No response received from MCP server", "raw_output": ""}
        
        # Close the process
        process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            
        return response
    except Exception as e:
        print(f"Error running MCP command: {e}")
        return {"error": str(e)}


def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    
    response = run_mcp_command("check_health")
    print(f"Response: {json.dumps(response, indent=2)}")
    
    return response


def test_upsert_memory():
    """Test upserting a memory item"""
    print("\n=== Testing Upsert Memory ===")
    
    # Generate a unique ID for this test
    memory_id = f"test_{uuid.uuid4().hex[:8]}"
    
    # Create test memory content
    memory_content = f"This is a test memory created at {time.ctime()}"
    
    # Create metadata
    metadata = {
        "source": "test_script",
        "timestamp": time.time(),
        "test": True
    }
    
    # Call the upsert tool
    response = run_mcp_command("upsert_memory", {
        "content": memory_content,
        "id": memory_id,
        "metadata": metadata
    })
    
    print(f"Response: {json.dumps(response, indent=2)}")
    
    return memory_id, response


def test_query_memory(query_text):
    """Test querying memory"""
    print(f"\n=== Testing Query Memory: '{query_text}' ===")
    
    response = run_mcp_command("query_memory", {
        "query": query_text
    })
    
    print(f"Response: {json.dumps(response, indent=2)}")
    
    return response


def test_delete_memory(memory_id):
    """Test deleting a memory item"""
    print(f"\n=== Testing Delete Memory: '{memory_id}' ===")
    
    response = run_mcp_command("delete_memory", {
        "memory_id": memory_id
    })
    
    print(f"Response: {json.dumps(response, indent=2)}")
    
    return response


def run_all_tests():
    """Run all MCP tests"""
    # Check the health of the server
    health_response = test_health_check()
    
    # If health check failed, stop testing
    if "error" in health_response:
        print("Health check failed, stopping tests.")
        return
    
    # Add some test data
    memory_id, upsert_response = test_upsert_memory()
    
    # Query for the data we just added
    query_response = test_query_memory("test memory")
    
    # Delete the test data
    delete_response = test_delete_memory(memory_id)
    
    # Query again to verify deletion
    query_after_delete = test_query_memory("test memory")
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    print("Starting MCP tool tests...")
    run_all_tests()
