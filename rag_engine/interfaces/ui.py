"""
Web UI implementation using Streamlit and Gradio.
"""
import streamlit as st
import os
from typing import Optional
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.loader import load_config


class StreamlitUI:
    """Streamlit-based web interface for RAG Engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.pipeline = None
        self._setup_page_config()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="RAG Engine",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🤖 RAG Engine")
        st.markdown("*Modular Retrieval-Augmented Generation Framework*")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Config file upload or path
            config_option = st.radio(
                "Configuration Source:",
                ["Upload File", "File Path"]
            )
            
            if config_option == "Upload File":
                uploaded_config = st.file_uploader(
                    "Upload config file",
                    type=['yml', 'yaml', 'json'],
                    help="Upload a YAML or JSON configuration file"
                )
                if uploaded_config:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_config.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_config.read())
                    self.config_path = temp_path
            else:
                config_path_input = st.text_input(
                    "Config file path:",
                    value=self.config_path or "",
                    help="Path to your configuration file"
                )
                if config_path_input:
                    self.config_path = config_path_input
            
            # Load configuration
            if self.config_path and os.path.exists(self.config_path):
                if st.button("🔄 Load Configuration"):
                    self._load_pipeline()
            
            # Pipeline status
            st.header("📊 Status")
            if self.pipeline:
                st.success("✅ Pipeline loaded")
                if hasattr(self.pipeline, '_is_built') and self.pipeline._is_built:
                    st.success("✅ Pipeline built")
                else:
                    if st.button("🔨 Build Pipeline"):
                        self._build_pipeline()
            else:
                st.warning("⚠️ No pipeline loaded")
        
        # Main content area
        if self.pipeline:
            self._render_main_interface()
        else:
            self._render_welcome_screen()
    
    def _load_pipeline(self):
        """Load the RAG pipeline from configuration."""
        try:
            with st.spinner("Loading configuration..."):
                config = load_config(self.config_path)
                self.pipeline = Pipeline(config)
                st.success("Configuration loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load configuration: {str(e)}")
    
    def _build_pipeline(self):
        """Build the RAG pipeline."""
        try:
            with st.spinner("Building pipeline..."):
                self.pipeline.build()
                st.success("Pipeline built successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to build pipeline: {str(e)}")
    
    def _render_welcome_screen(self):
        """Render welcome screen when no pipeline is loaded."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to RAG Engine! 🚀
            
            To get started:
            
            1. **Upload or specify** a configuration file in the sidebar
            2. **Load** the configuration
            3. **Build** the pipeline
            4. **Start chatting** with your documents!
            
            ### Example Configuration:
            
            ```yaml
            documents:
              - type: txt
                path: ./documents/sample.txt
            
            chunking:
              method: fixed
              max_tokens: 256
              overlap: 20
            
            embedding:
              provider: huggingface
              model: sentence-transformers/all-MiniLM-L6-v2
            
            vectorstore:
              provider: chroma
              persist_directory: ./vector_store
            
            retrieval:
              top_k: 3
            
            prompting:
              system_prompt: "You are a helpful assistant."
            
            llm:
              provider: openai
              model: gpt-3.5-turbo
              temperature: 0.7
              api_key: ${OPENAI_API_KEY}
            
            output:
              method: console
            ```
            """)
    
    def _render_main_interface(self):
        """Render the main chat interface."""
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📊 Analytics"])
        
        with tab1:
            self._render_chat_interface()
        
        with tab2:
            self._render_documents_interface()
        
        with tab3:
            self._render_analytics_interface()
    
    def _render_chat_interface(self):
        """Render the chat interface."""
        st.header("💬 Chat with Your Documents")
        
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = self.pipeline.chat(prompt)
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            error_msg = "Sorry, I couldn't generate a response. Please check your configuration."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    def _render_documents_interface(self):
        """Render the documents management interface."""
        st.header("📄 Document Management")
        
        if hasattr(self.pipeline, 'documents') and self.pipeline.documents:
            st.subheader("Loaded Documents")
            for i, doc in enumerate(self.pipeline.documents):
                with st.expander(f"Document {i+1}: {doc.get('path', 'Unknown')}"):
                    st.write(f"**Type:** {doc.get('type', 'Unknown')}")
                    st.write(f"**Size:** {len(doc.get('content', ''))} characters")
                    if 'metadata' in doc:
                        st.write(f"**Metadata:** {doc['metadata']}")
                    
                    # Show content preview
                    content = doc.get('content', '')
                    if content:
                        st.text_area(
                            "Content Preview:",
                            content[:500] + "..." if len(content) > 500 else content,
                            height=100,
                            disabled=True
                        )
        
        if hasattr(self.pipeline, 'chunks') and self.pipeline.chunks:
            st.subheader("Document Chunks")
            st.write(f"Total chunks: {len(self.pipeline.chunks)}")
            
            # Chunk viewer
            chunk_index = st.selectbox("Select chunk to view:", range(len(self.pipeline.chunks)))
            if chunk_index is not None:
                chunk = self.pipeline.chunks[chunk_index]
                with st.expander(f"Chunk {chunk_index + 1}"):
                    st.write(f"**Content:** {chunk.get('content', 'No content')}")
                    if 'metadata' in chunk:
                        st.write(f"**Metadata:** {chunk['metadata']}")
    
    def _render_analytics_interface(self):
        """Render analytics and configuration details."""
        st.header("📊 Analytics & Configuration")
        
        if self.pipeline and self.pipeline.config:
            config = self.pipeline.config
            
            # Configuration summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 Configuration")
                st.write(f"**Documents:** {len(config.documents)} configured")
                st.write(f"**Chunking:** {config.chunking.method}")
                st.write(f"**Embedding:** {config.embedding.provider}")
                st.write(f"**Vector Store:** {config.vectorstore.provider}")
                st.write(f"**LLM:** {config.llm.provider}")
            
            with col2:
                st.subheader("📈 Statistics")
                if hasattr(self.pipeline, 'documents'):
                    st.write(f"**Loaded Documents:** {len(self.pipeline.documents)}")
                if hasattr(self.pipeline, 'chunks'):
                    st.write(f"**Total Chunks:** {len(self.pipeline.chunks)}")
                st.write(f"**Retrieval Top-K:** {config.retrieval.top_k}")
                st.write(f"**Max Tokens:** {config.chunking.max_tokens}")
            
            # Full configuration display
            with st.expander("🔍 Full Configuration"):
                st.json(config.dict())


def run_streamlit_ui(config_path: Optional[str] = None):
    """Run the Streamlit UI."""
    ui = StreamlitUI(config_path)
    ui.run()


# Gradio interface (alternative to Streamlit)
def create_gradio_interface(config_path: Optional[str] = None):
    """Create a Gradio interface for RAG Engine."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio")
    
    # Load pipeline
    pipeline = None
    if config_path and os.path.exists(config_path):
        try:
            config = load_config(config_path)
            pipeline = Pipeline(config)
            pipeline.build()
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
    
    def chat_function(message, history):
        """Handle chat messages in Gradio."""
        if not pipeline:
            return "❌ Pipeline not loaded. Please check configuration."
        
        try:
            response = pipeline.chat(message)
            return response or "Sorry, I couldn't generate a response."
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        fn=chat_function,
        title="🤖 RAG Engine",
        description="Chat with your documents using Retrieval-Augmented Generation",
        theme=gr.themes.Soft(),
        examples=[
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key findings?"
        ]
    )
    
    return iface


def run_gradio_ui(config_path: Optional[str] = None, **kwargs):
    """Run the Gradio UI."""
    iface = create_gradio_interface(config_path)
    iface.launch(**kwargs)
