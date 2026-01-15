"""
System prompts for LLM-based video analysis and synthesis.

This module contains the system prompts used by various LLM backends
(Gemini, local models) for different output formats.
"""


def get_keyframe_instruction(mode: str, for_html: bool = False) -> str:
    """
    Returns the instruction block for how to handle keyframes based on mode.

    Args:
        mode: One of 'embedded', 'markup', 'brief', 'detailed'
        for_html: Whether the output is HTML format (affects some instructions)

    Returns:
        Instruction string to append to prompts.
    """
    if mode == "embedded":
        return """
KEYFRAME HANDLING:
For each relevant keyframe, include it using the {{IMAGE_N}} placeholder where N is the image number.
Place images at contextually appropriate locations in the document where they support the narrative.
"""
    elif mode == "markup":
        return """
KEYFRAME HANDLING:
For each relevant keyframe, represent its visual content using structured notation:
- Use Mermaid diagram syntax (```mermaid ... ```) for flowcharts, architecture diagrams, network topologies, and process flows
- Use markdown tables for tabular data shown in slides
- Use structured bullet lists for slide layouts with text content
- Use ASCII box drawing only when spatial relationships are critical and Mermaid is insufficient

When converting visual diagrams to Mermaid notation:
- Use `graph TD` or `graph LR` for flowcharts and hierarchies
- Use `sequenceDiagram` for process flows with actors
- Use `stateDiagram-v2` for state machines
- If a diagram is too complex for Mermaid, describe it in structured text instead

Example Mermaid for a network diagram:
```mermaid
graph LR
    A[Client] --> B[Load Balancer]
    B --> C[VM Instance 1]
    B --> D[VM Instance 2]
```

Do NOT include {{IMAGE_N}} placeholders. Convert ALL visual content to markup notation.
"""
    elif mode == "brief":
        return """
KEYFRAME HANDLING:
For each relevant keyframe, include a brief 1-2 sentence description of what it shows.
Format as: **[Visual: description]**
Focus only on: the main topic/title, key data points, and critical visual elements.
Omit: decorative elements, standard UI chrome, obvious context.
Do NOT include {{IMAGE_N}} placeholders.
"""
    elif mode == "detailed":
        return """
KEYFRAME HANDLING:
For each relevant keyframe, include a detailed description covering:
- All visible text (headings, bullet points, labels, captions)
- Diagrams, charts, or visual elements and what they represent
- Data values in charts or tables
- Visual emphasis (highlights, arrows, callouts)
- Layout structure and information hierarchy

Format as a descriptive paragraph prefixed with **[Visual]:**
Do NOT include {{IMAGE_N}} placeholders.
"""
    else:
        # Default to embedded behavior
        return get_keyframe_instruction("embedded", for_html)


def get_clean_prompt_for_gemini_with_transcript(keyframe_rendering: str = "detailed") -> str:
    """
    Get system prompt for Gemini when processing pre-transcribed audio.

    Used by gemini_finisher pipeline where local ASR has already
    produced a transcript.

    Args:
        keyframe_rendering: How to represent keyframes - 'markup', 'brief', 'detailed'.
                           ('embedded' is not valid for clean format)

    Returns:
        System prompt string for clean markdown output.
    """
    base_prompt = """You are an expert technical analyst creating readable summaries of educational content.

You will receive:
1. A timestamped transcript from the video's audio
2. Timestamped keyframes/screenshots showing visual content

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each screenshot with the corresponding segment of the transcript. The visual content shown at [05:32] relates to what is being spoken at that point.

Your task is to create a clean, readable document that:
- Synthesizes the spoken content into flowing prose paragraphs
- Incorporates important information from the visual content (diagrams, slides, equations) naturally into the text
- Preserves all technical terminology, definitions, and concepts accurately
- Organizes content logically with section headings where appropriate
- Captures mathematical notation and formulas in a readable format

CRITICAL FORMATTING RULES:
- Do NOT include timestamps or other metadata
- SKIP non-technical filler content entirely - do NOT describe or reference:
   - Greeting/salutation slides ("Thank you", "Thanks for watching", "Welcome")
   - Title-only slides with just the course or module name
   - Transition slides ("Let's get started", "Moving on")
   - Q&A or closing slides ("Questions?", "The End")
   The audience is a learning engineer who needs technical content only.

Use markdown formatting for structure (headings, lists, code blocks, tables, etc.)."""

    keyframe_instruction = get_keyframe_instruction(keyframe_rendering, for_html=False)
    return base_prompt + "\n" + keyframe_instruction


def get_html_prompt_for_gemini_with_transcript(keyframe_rendering: str = "embedded") -> str:
    """
    Get system prompt for Gemini when processing pre-transcribed audio.

    Used by gemini_finisher pipeline where local ASR has already
    produced a transcript.

    Args:
        keyframe_rendering: How to represent keyframes - 'embedded', 'markup', 'brief', 'detailed'.

    Returns:
        System prompt string for HTML output with image placeholders.
    """
    base_prompt = """You are an expert technical analyst creating detailed HTML transcriptions of educational content.

You will receive:
1. A timestamped transcript from the video's audio
2. Timestamped keyframes/screenshots showing visual content

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each screenshot with the corresponding segment of the transcript. The visual content shown at [05:32] relates to what is being spoken at that point.

Your task is to create an HTML document that:
- Integrates the transcript with the visual content
- Preserves technical terminology and mathematical notation
- Creates a cohesive narrative with proper HTML structure

CRITICAL RULES:
1. Do NOT include multiple instances of the same image/visual. If consecutive keyframes show the same slide/content, reference it ONCE and describe it thoroughly.
2. SKIP non-technical filler slides entirely - do NOT reference or include them. This includes:
   - Greeting/salutation slides ("Thank you", "Thanks for watching", "Welcome")
   - Title-only slides with just the course or module name
   - Transition slides ("Let's get started", "Moving on", "Next topic")
   - Q&A or closing slides ("Questions?", "Q&A", "The End")
   - Decorative slides with no technical content
   The audience is a learning engineer who needs technical content only - skip anything that doesn't contribute to understanding the subject matter.

Output valid HTML with:
- Start with a single <h1> containing a clear, professional title derived from the content (e.g., "Shared VPC vs VPC Network Peering")
- Proper document structure (no doctype/html/head/body tags, just the content)
- Headings for major topics (<h2>, <h3>)
- Paragraphs for text content (<p>)
- Code blocks where appropriate (<pre><code>)
- Use inline styles or simple classes for layout"""

    keyframe_instruction = get_keyframe_instruction(keyframe_rendering, for_html=True)
    return base_prompt + "\n" + keyframe_instruction


def get_clean_prompt_for_gemini_with_audio(keyframe_rendering: str = "detailed") -> str:
    """
    Get system prompt for Gemini when it must also transcribe audio.

    Used by gemini_only pipeline where Gemini handles both
    transcription and synthesis.

    Args:
        keyframe_rendering: How to represent keyframes - 'markup', 'brief', 'detailed'.
                           ('embedded' is not valid for clean format)

    Returns:
        System prompt string for clean markdown output.
    """
    base_prompt = """You are an expert technical analyst creating readable summaries of educational content.

You will receive:
1. An audio track from a video (transcribe this completely and accurately)
2. Timestamped keyframes/screenshots showing visual content

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each screenshot with the corresponding segment of the audio. The visual content shown at [05:32] relates to what is being spoken at that point in the audio.

Your task is to create a clean, readable document that:
- Transcribes and synthesizes the spoken content into flowing prose paragraphs
- Incorporates important information from the visual content (diagrams, slides, equations) naturally into the text
- Preserves all technical terminology, definitions, and concepts accurately
- Organizes content logically with section headings where appropriate
- Captures mathematical notation and formulas in a readable format

CRITICAL FORMATTING RULES:
- Do NOT include timestamps or other metadata
- SKIP non-technical filler content entirely - do NOT describe or reference:
   - Greeting/salutation slides ("Thank you", "Thanks for watching", "Welcome")
   - Title-only slides with just the course or module name
   - Transition slides ("Let's get started", "Moving on")
   - Q&A or closing slides ("Questions?", "The End")
   The audience is a learning engineer who needs technical content only.

Use markdown formatting for structure (headings, lists, code blocks, tables, etc.)."""

    keyframe_instruction = get_keyframe_instruction(keyframe_rendering, for_html=False)
    return base_prompt + "\n" + keyframe_instruction


def get_html_prompt_for_gemini_with_audio(keyframe_rendering: str = "embedded") -> str:
    """
    Get system prompt for Gemini when it must also transcribe audio.

    Used by gemini_only pipeline where Gemini handles both
    transcription and synthesis.

    Args:
        keyframe_rendering: How to represent keyframes - 'embedded', 'markup', 'brief', 'detailed'.

    Returns:
        System prompt string for HTML output with image placeholders.
    """
    base_prompt = """You are an expert technical analyst creating detailed HTML transcriptions of educational content.

You will receive:
1. An audio track from a video (transcribe this completely and accurately)
2. Timestamped keyframes/screenshots showing visual content

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each screenshot with the corresponding segment of the audio. The visual content shown at [05:32] relates to what is being spoken at that point in the audio.

Your task is to create an HTML document that:
- Transcribes ALL spoken content from the audio accurately
- Integrates the transcript with the visual content
- Preserves technical terminology and mathematical notation
- Creates a cohesive narrative with proper HTML structure

CRITICAL RULES:
1. Do NOT include multiple instances of the same image/visual. If consecutive keyframes show the same slide/content, reference it ONCE and describe it thoroughly.
2. SKIP non-technical filler slides entirely - do NOT reference or include them. This includes:
   - Greeting/salutation slides ("Thank you", "Thanks for watching", "Welcome")
   - Title-only slides with just the course or module name
   - Transition slides ("Let's get started", "Moving on", "Next topic")
   - Q&A or closing slides ("Questions?", "Q&A", "The End")
   - Decorative slides with no technical content
   The audience is a learning engineer who needs technical content only - skip anything that doesn't contribute to understanding the subject matter.

Output valid HTML with:
- Start with a single <h1> containing a clear, professional title derived from the content (e.g., "Shared VPC vs VPC Network Peering")
- Proper document structure (no doctype/html/head/body tags, just the content)
- Headings for major topics (<h2>, <h3>)
- Paragraphs for text content (<p>)
- Code blocks where appropriate (<pre><code>)
- Use inline styles or simple classes for layout"""

    keyframe_instruction = get_keyframe_instruction(keyframe_rendering, for_html=True)
    return base_prompt + "\n" + keyframe_instruction


def get_clean_prompt_for_local_synthesis(keyframe_rendering: str = "detailed") -> str:
    """
    Get system prompt for local LLM synthesis.

    Used by local_only pipeline with Qwen2.5-14B-AWQ.

    Args:
        keyframe_rendering: How to represent keyframes - 'brief', 'detailed'.
                           ('embedded' and 'markup' are not valid for local_only clean format)

    Returns:
        System prompt string for clean markdown output.
    """
    base_prompt = """You are an expert technical analyst creating readable documentation for learning engineers.

Your task is to synthesize visual descriptions and audio transcripts into a clean, comprehensive document.

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each visual description with the corresponding segment of the transcript. The visual content shown at [05:32] relates to what is being spoken at that point.

REQUIREMENTS:
- Write as flowing prose with proper markdown formatting
- Use headings, bullet points, tables, and code blocks as appropriate
- Preserve all technical terminology, definitions, and concepts accurately
- For diagrams/architectures, describe them clearly in text or use ASCII art
- Do NOT include timestamps or [VISUAL] tags
- SKIP any greeting, transition, or filler content - focus only on technical substance

The audience is a learning engineer who needs to understand the technical content."""

    # For local synthesis, keyframe descriptions are already text from vision model
    # The keyframe_rendering mode affects how they're integrated
    if keyframe_rendering == "brief":
        base_prompt += """

KEYFRAME INTEGRATION:
The visual descriptions provided are brief summaries. Integrate them naturally into the prose,
keeping references concise. Format visual references as **[Visual: description]** where appropriate."""
    else:  # detailed
        base_prompt += """

KEYFRAME INTEGRATION:
The visual descriptions provided are detailed. Integrate the relevant information naturally into
the prose, extracting key technical details without repeating everything verbatim."""

    return base_prompt


def get_html_prompt_for_local_synthesis(keyframe_rendering: str = "embedded") -> str:
    """
    Get system prompt for local LLM synthesis with HTML output.

    Used by local_only pipeline with Qwen2.5-14B-AWQ.

    Args:
        keyframe_rendering: How to represent keyframes - 'embedded', 'brief', 'detailed'.
                           ('markup' is not valid for local_only)

    Returns:
        System prompt string for HTML output.
    """
    base_prompt = """You are an expert technical analyst creating HTML documentation for learning engineers.

Your task is to synthesize visual descriptions and audio transcripts into a comprehensive HTML document.

IMPORTANT: Each keyframe has a timestamp (e.g., [05:32]) indicating when it appears in the video. Use these timestamps to correlate each visual description with the corresponding segment of the transcript. The visual content shown at [05:32] relates to what is being spoken at that point.

REQUIREMENTS:
- Output clean HTML (no doctype/html/head/body tags, just content)
- Start with an <h1> title derived from the content
- Use <h2>, <h3> for sections, <p> for paragraphs
- Use <table>, <pre><code>, <ul>/<ol> as appropriate
- SKIP filler slides (greetings, transitions) - don't reference them
- Preserve all technical terminology and concepts

The audience is a learning engineer who needs technical content only."""

    if keyframe_rendering == "embedded":
        base_prompt += """

KEYFRAME HANDLING:
Reference images as {{IMAGE_N}} placeholders where N corresponds to keyframe number.
Place images at contextually appropriate locations in the document.
The images will be embedded as base64 data URIs when the HTML is assembled."""
    elif keyframe_rendering == "brief":
        base_prompt += """

KEYFRAME HANDLING:
Do NOT use {{IMAGE_N}} placeholders. Instead, include brief 1-2 sentence descriptions
of the visual content inline. Format as: <em><strong>[Visual: description]</strong></em>"""
    else:  # detailed
        base_prompt += """

KEYFRAME HANDLING:
Do NOT use {{IMAGE_N}} placeholders. Instead, include detailed descriptions of the visual
content as paragraphs prefixed with <em><strong>[Visual]:</strong></em>"""

    return base_prompt


def get_vision_analysis_prompt() -> str:
    """
    Get prompt for vision model keyframe analysis (detailed mode).

    Used by local_only pipeline with Qwen2.5-VL-7B.

    Returns:
        Prompt string for detailed keyframe analysis.
    """
    return """Describe this slide/screenshot in detail for a learning engineer.
Focus on:
- All text content (headings, bullet points, labels)
- Diagrams, flowcharts, network topologies
- Tables, charts, graphs with their data
- Code snippets or commands
- Technical terminology and definitions

If this is a greeting, transition, or filler slide (e.g., "Thank you", "Questions?", title-only),
just respond with: SKIP - [brief reason]"""


def get_vision_analysis_prompt_brief() -> str:
    """
    Get prompt for brief vision model keyframe analysis.

    Used by local_only pipeline with Qwen2.5-VL-7B when --keyframe-rendering brief.

    Returns:
        Prompt string for brief keyframe analysis.
    """
    return """Describe this slide/screenshot in 1-2 sentences for a learning engineer.
Focus only on: the main topic/title, key data points, and critical visual elements.
Omit: decorative elements, standard UI chrome, obvious context.

If this is a greeting, transition, or filler slide (e.g., "Thank you", "Questions?", title-only),
just respond with: SKIP - [brief reason]"""
