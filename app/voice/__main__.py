"""Launch the LiveKit worker with python -m app.voice."""

from livekit.agents import cli

from app.voice.agent import server

if __name__ == "__main__":
    cli.run_app(server)
