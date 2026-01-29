import streamlit as st

from langgraph_database import chatbot,gen_model,rec_all_thread
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid
import os
st.set_page_config(page_title="LangGraph Chatbot")

os.environ['LANGCHAIN_PROJECT']='Langgraph chatbot'

# --------------------Utility functions ------------------------------------------
def gen_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = gen_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread(thread_id):

    if thread_id  not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

from langchain_core.messages import HumanMessage, AIMessage

def load_conv(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    cleaned_messages = []

    for msg in messages:
        # ✅ Keep user messages
        if isinstance(msg, HumanMessage):
            cleaned_messages.append(msg)

        # ✅ Keep assistant messages with real text only
        elif isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                cleaned_messages.append(msg)

        # ❌ Skip ToolMessage, SystemMessage, etc.
        else:
            continue

    return cleaned_messages

def generate_title(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    if not messages:
        return "New Chat"

    convo_text = "\n".join(
        f"{type(m).__name__}: {m.content}"
        for m in messages[:6]
    )

    prompt = f"""
    Generate a short (3–6 words) descriptive title for this conversation.
    Do not use quotes.

    Conversation:
    {convo_text}
    """

    return gen_model.invoke(prompt).content.strip()
#------------------------Session Initialization--------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = gen_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = rec_all_thread()

if "chat_title" not in st.session_state:
    st.session_state["chat_title"] = {}   # thread_id → title

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

add_thread(st.session_state["thread_id"])

# --------------------Sidebar UI ------------------------------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My conversations")

for thread_id in st.session_state['chat_thread'][::-1]:
    title = st.session_state["chat_title"].get(thread_id,"Current chat")

    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id']=thread_id
        messages=load_conv(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role':role,'content':msg.content})

        st.session_state['message_history'] =temp_messages





# st.session_state  -> dict does not change on enter press


#---------------------------------------------------------------------

config1= {"configurable":{"thread_id":st.session_state['thread_id']}}

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    

    
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config1,
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

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
