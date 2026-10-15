import streamlit as st

from src.config.settings import validate_runtime_configuration
from src.ui.dashboard import render_dashboard


def main() -> None:
    try:
        validate_runtime_configuration()
    except RuntimeError as exc:
        st.error(str(exc))
        st.info("Set MODEL_PROVIDER, MODEL_NAME, and API_KEY in the local .env file before starting the app.")
        st.stop()

    render_dashboard()


if __name__ == "__main__":
    main()
