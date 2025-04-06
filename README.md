# LEAP Agent for Climate Research

This fork of the Agent-First-Organization repository includes several enhancements and additions focused on climate research applications.

## Added Features

### LEAP Agent
- Added specialized LEAP agent for climate research applications
- Enhanced agent capabilities for scientific research and collaboration

### Configuration Files
- Added `config.json` - Base configuration for the LEAP agent
- Added `leap_enhanced_config.json` - Extended configuration with additional parameters for advanced use cases

### Evaluation Framework
- Implemented comprehensive evaluation capabilities
- Added metrics tracking for agent performance
- Includes goal completion rate tracking

### TaskGraphs
- Added enhanced TaskGraph functionality
- Improved task planning and execution
- Better coordination between specialized workers

## Specialized Workers
Added new workers for climate research:
- Citation generator for academic references
- Dataset finder for locating relevant climate data
- Research collaboration assistant

## Getting Started
To use these enhancements:
1. Clone this repository
2. Install dependencies with `pip install -r requirements.txt`
3. Configure your API keys in a new `.env` file
4. Use the LEAP configuration files to initialize the agent
5. Run the agent with `python run.py --input-dir ./examples/leap`