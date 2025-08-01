#!/usr/bin/env python3
"""
Server startup script for Document AI Agent.
Handles initialization, health checks, and graceful shutdown.
"""

import os
import sys
import signal
import logging
import asyncio
from pathlib import Path

import uvicorn
from config import HOST, PORT, DEBUG, DATA_DIR, UPLOADS_DIR, TEMPLATES_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('document_ai_agent.log')
    ]
)

logger = logging.getLogger(__name__)

class ServerManager:
    """Manages the FastAPI server lifecycle."""

    def __init__(self):
        self.server = None
        self.shutdown_event = asyncio.Event()

    def setup_directories(self):
        """Ensure all required directories exist."""
        directories = [DATA_DIR, UPLOADS_DIR, TEMPLATES_DIR]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ready: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                sys.exit(1)

    def check_dependencies(self):
        """Check if all required dependencies are available."""
        required_packages = [
            'fastapi', 'uvicorn', 'langchain', 'transformers',
            'torch', 'faiss-cpu', 'sentence-transformers'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install missing packages using: pip install -r requirements.txt")
            sys.exit(1)

        logger.info("All dependencies are available")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def print_startup_info(self):
        """Print startup information."""
        print("\n" + "="*60)
        print("🚀 Document AI Agent Server")
        print("="*60)
        print(f"📡 Server URL: http://{HOST}:{PORT}")
        print(f"📚 API Docs: http://{HOST}:{PORT}/docs")
        print(f"🔍 Health Check: http://{HOST}:{PORT}/health")
        print(f"📁 Data Directory: {DATA_DIR}")
        print(f"📤 Upload Directory: {UPLOADS_DIR}")
        print(f"🐛 Debug Mode: {DEBUG}")
        print("="*60)
        print("💡 Features:")
        print("   • Multi-format document support (PDF, DOCX, TXT)")
        print("   • Advanced vector search with FAISS")
        print("   • BAAI/bge-large-en-v1.5 embeddings")
        print("   • DialoGPT-large for response generation")
        print("   • LangChain integration for advanced workflows")
        print("   • Confidence scoring and source attribution")
        print("="*60)
        print("Ready to process your documents! 🎯\n")

    async def health_check(self):
        """Perform initial health check."""
        try:
            # Import main app to trigger initialization
            from main import app, ai_agent

            # Check if AI agent is ready
            stats = ai_agent.get_stats()
            logger.info(f"AI Agent initialized - {stats}")

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def run_server(self):
        """Run the FastAPI server."""
        config = uvicorn.Config(
            "main:app",
            host=HOST,
            port=PORT,
            reload=DEBUG,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )

        self.server = uvicorn.Server(config)

        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

    async def main(self):
        """Main async entry point."""
        logger.info("Starting Document AI Agent Server...")

        # Setup
        self.setup_directories()
        self.check_dependencies()
        self.setup_signal_handlers()

        # Health check
        if not await self.health_check():
            logger.error("Health check failed, exiting...")
            sys.exit(1)

        # Print startup info
        self.print_startup_info()

        # Run server
        try:
            server_task = asyncio.create_task(self.run_server())

            # Wait for shutdown signal or server completion
            done, pending = await asyncio.wait(
                [server_task, asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Graceful shutdown
            if self.server:
                logger.info("Shutting down server...")
                self.server.should_exit = True

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            logger.info("Server shutdown complete")

def main():
    """Entry point for the application."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("Error: Python 3.8 or higher is required")
            sys.exit(1)

        # Create and run server manager
        server_manager = ServerManager()
        asyncio.run(server_manager.main())

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
