# Temporary database disable to fix symbol formatting issues
# This will allow the app to work without database connection issues

import main_app

# Replace database initialization
original_initialize = main_app.initialize_session_state

def patched_initialize():
    # Call original but skip database
    original_initialize()
    # Force disable database
    main_app.st.session_state.db_manager = None
    main_app.st.session_state.db_connected = False
    main_app.st.session_state.performance_analyzer = None
    main_app.st.session_state.auto_trader = None
    main_app.st.session_state.live_tracker = None

main_app.initialize_session_state = patched_initialize