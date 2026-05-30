import streamlit as st
from langgraph_backend import chatbot, get_all_threads
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def get_configuration(thread_id=None):
    if thread_id is None:
        thread_id = st.session_state['thread_id']
    return {'configurable': {'thread_id': thread_id}, 'run_name': 'chat_turn'}

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []
    st.session_state['pending_interrupt'] = None

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_thread(thread_id):
    state = chatbot.get_state(config=get_configuration(thread_id))
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def get_pending_interrupt(thread_id=None):
    """Return the interrupt prompt value if the graph is paused, else None."""
    state = chatbot.get_state(config=get_configuration(thread_id))
    for task in state.tasks:
        if task.interrupts:
            # interrupt() was called inside a tool; surface its value
            return task.interrupts[0].value
    return None

def stream_graph_response(payload):
    """
    Stream the assistant response for a given payload.

    payload is either a new turn ({"messages": [HumanMessage(...)]})
    or a resume command (Command(resume=...)) used to continue a paused graph.
    Returns the assistant text that was streamed.
    """
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                payload,
                config=get_configuration(),
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message (it may be empty if the turn only triggered a tool)
    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

    # After streaming, check whether the graph paused waiting for human input
    st.session_state['pending_interrupt'] = get_pending_interrupt()
    return ai_message


# ****************************** Session State Management ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'pending_interrupt' not in st.session_state:
    st.session_state['pending_interrupt'] = None

add_thread(st.session_state['thread_id'])

# ****************************** Sidebar UI ******************************
st.sidebar.title("Omnichat")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Chat Threads")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_thread(thread_id)

        temp_message_history = []
        for message in messages:
            temp_message_history.append({'role': message.type, 'content': message.content})

        st.session_state['message_history'] = temp_message_history
        # Restore any pending human-in-the-loop interrupt for this thread
        st.session_state['pending_interrupt'] = get_pending_interrupt(thread_id)

# ****************************** Main UI ******************************
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

# ****************************** Human-in-the-loop approval ******************************
if st.session_state.get('pending_interrupt') is not None:
    with st.chat_message("assistant"):
        st.warning(f"⚠️ **Human approval required**\n\n{st.session_state['pending_interrupt']}")

    col1, col2 = st.columns(2)
    approve = col1.button("✅ Approve", use_container_width=True)
    decline = col2.button("❌ Decline", use_container_width=True)

    if approve or decline:
        decision = "yes" if approve else "no"
        st.session_state['pending_interrupt'] = None
        # Resume the paused graph with the human's decision
        stream_graph_response(Command(resume=decision))
        st.rerun()

user_input = st.chat_input(
    'Type here',
    disabled=st.session_state.get('pending_interrupt') is not None,
)

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Assistant streaming block
    stream_graph_response({"messages": [HumanMessage(content=user_input)]})

    # If the turn paused for human approval, rerun to render the approval UI
    if st.session_state.get('pending_interrupt') is not None:
        st.rerun()
