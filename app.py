import os
import time
import importlib.util
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random


def import_clean_text():
    # dynamic import of scripts/clean_text.py to reuse clean_text()
    mod_path = os.path.join(os.path.dirname(__file__), 'scripts', 'clean_text.py')
    spec = importlib.util.spec_from_file_location('clean_text_module', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.clean_text


@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, 'models')
    vec = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    clf = joblib.load(os.path.join(models_dir, 'logreg_tfidf.joblib'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
    return vec, clf, le


def predict_and_explain(text, vec, clf, le, clean_fn, topk=8):
    cleaned = clean_fn(text)
    X = vec.transform([cleaned])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]

    # explain via coef * tfidf
    try:
        feature_names = vec.get_feature_names_out()
    except Exception:
        feature_names = vec.get_feature_names()

    # compute contributions only for features present in this input
    tf = X.toarray().ravel()
    nz = np.where(tf != 0)[0]
    top_features = []
    if nz.size > 0:
        coef = clf.coef_[pred_idx]
        # contribution = coef * tfidf value
        contrib_full = coef * tf
        # select non-zero feature contributions and sort descending
        nz_contrib = contrib_full[nz]
        order = np.argsort(nz_contrib)[::-1]
        chosen = nz[order][:topk]
        for i in chosen:
            # only include if contribution is non-negligible
            if abs(float(contrib_full[i])) > 1e-12:
                top_features.append((feature_names[i], float(contrib_full[i])))

    return pred_label, probs, top_features


def combine_wellness_score(pred_label, confidence, sleep_hours, stress_level, mood_choice):
    """Combine text-based prediction + self-reported wellness signals into a 0-100 wellness score.

    Returns: (wellness_score, subscores_dict)
    """
    # Adjustable weights
    W_TEXT = 0.40
    W_SLEEP = 0.20
    W_STRESS = 0.20
    W_MOOD = 0.20

    # Map category spectrum (same as emotion_map used in the chart)
    emotion_map = {
        'Normal': 6.0,
        'Stress': 4.0,
        'Anxiety': 3.0,
        'Bipolar': 3.5,
        'Depression': 2.0,
        'Personality disorder': 2.5,
        'Suicidal': 1.0,
    }
    # convert category value (1..6) to 0..100 (Suicidal=0, Normal=100)
    cat_val = emotion_map.get(pred_label, 3.0)
    text_category_score = (cat_val - 1.0) / (6.0 - 1.0) * 100.0

    # Adjust by confidence: if confidence is low, pull toward neutral (50)
    text_subscore = float(confidence) * text_category_score + (1.0 - float(confidence)) * 50.0

    # Sleep score: optimal 7-9 -> 100; degrade linearly outside that band
    h = float(sleep_hours)
    if 7.0 <= h <= 9.0:
        sleep_subscore = 100.0
    elif h < 7.0:
        # linear through (4h -> 40) and (7h -> 100)
        sleep_subscore = 40.0 + (h - 4.0) * 20.0
    else:
        # h > 9, linear through (9h -> 100) and (12h -> 60)
        sleep_subscore = 100.0 + (h - 9.0) * ((60.0 - 100.0) / 3.0)
    sleep_subscore = float(max(0.0, min(100.0, sleep_subscore)))

    # Stress score: here `stress_level` represents calmness (0-10), higher is better
    lvl = float(stress_level)
    stress_subscore = max(0.0, min(100.0, (lvl / 10.0) * 100.0))

    # Mood check-in mapping
    mood_map = {
        '😊 Feeling good': 100.0,
        '🙂 Doing okay': 75.0,
        '😐 Just neutral': 50.0,
        '😔 Having a hard time': 25.0,
        '😰 Really struggling': 0.0,
    }
    mood_subscore = float(mood_map.get(mood_choice, 50.0))

    # Weighted combination
    wellness_score = (
        W_TEXT * text_subscore + W_SLEEP * sleep_subscore + W_STRESS * stress_subscore + W_MOOD * mood_subscore
    )

    subs = {
        'text': round(text_subscore, 2),
        'sleep': round(sleep_subscore, 2),
        'stress': round(stress_subscore, 2),
        'mood': round(mood_subscore, 2),
    }

    return float(max(0.0, min(100.0, wellness_score))), subs


def get_suggestions(wellness_score, subs):
    """Return 2-3 warm, non-clinical suggestions based on wellness and subscores.

    This function builds suggestions dynamically by checking each sub-score
    and picking a random phrasing from several options so outputs vary.
    """
    # Unpack subs
    text_s = subs.get('text', 50.0)
    sleep_s = subs.get('sleep', 50.0)
    stress_s = subs.get('stress', 50.0)
    mood_s = subs.get('mood', 50.0)

    # Early positive reinforcement
    if wellness_score > 75:
        return ["💡 You're doing well — keep noticing what's working for you."]

    tips = []

    # Candidate phrasings
    sleep_phrases = [
        "🌙 A short wind-down tonight may help—dim lights and step away from screens an hour before bed.",
        "🌙 Try a calming bedtime ritual: warm drink, soft music, and a predictable routine.",
        "🌙 If sleep is short, a brief afternoon rest or relaxation exercise could help recharge you.",
    ]

    stress_phrases = [
        "💆‍♀️ Try a 3-minute grounding break: breathe deeply and notice 3 things around you.",
        "💆‍♀️ A quick stretch and quiet breath can lower stress—give it 2–3 minutes.",
        "💆‍♀️ Step outside for a short walk, even 5 minutes can help shift perspective.",
    ]

    text_phrases = [
        "📝 If it helps, try writing for 5 minutes about what's most on your mind—no pressure, just noticing.",
        "📝 Try answering: 'What happened?', 'How did it feel?', 'What do I need right now?'.",
        "📝 Free-write for a few minutes; it can surface small next steps or easing thoughts.",
    ]

    mood_phrases = [
        "🌱 Name one small thing that felt okay today and notice it for a minute.",
        "🌱 If you're having a hard time, try a brief pause: 4 slow breaths and a hand on your heart.",
        "🌱 Consider a tiny self-care act now—sip water, step outside, or message a friend.",
    ]

    # Threshold checks (each subscore independent)
    if sleep_s < 60:
        tips.append(random.choice(sleep_phrases))
    if stress_s < 60:
        tips.append(random.choice(stress_phrases))
    if text_s < 60:
        tips.append(random.choice(text_phrases))
    if mood_s < 60:
        tips.append(random.choice(mood_phrases))

    # Very low overall -> add suggestion to reach out (soft)
    if wellness_score < 30:
        tips.append("🤝 If you're able, consider reaching out to someone you trust or a professional—you deserve support.")

    # If no specific tips were added, add a gentle default
    if not tips:
        tips = ["💡 Small, consistent steps often help—notice one positive thing today and celebrate it."]

    # Return up to 3 varied suggestions
    return tips[:3]


def main():
    st.set_page_config(page_title="SoulScript", page_icon="🌙", layout="centered")
    # Comprehensive dark theme CSS (force dark regardless of OS/browser preference)
    dark_css = """
    <style>
    /* Force dark color scheme and set background on core root containers */
    :root, html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stToolbar"] {
        background-color: #0f0f1e !important;
        color: #e6eef8 !important;
        color-scheme: dark !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    /* Ensure form controls follow dark styling */
    input, textarea, select, button, .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #1a1a2e !important;
        color: #e6eef8 !important;
        border-color: rgba(255,255,255,0.06) !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #6a5acd !important; /* muted purple */
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: none !important;
    }

    /* Textareas and inputs */
    textarea, .stTextArea>div>textarea, .stTextInput>div>input {
        background-color: #1a1a2e !important;
        color: #e6eef8 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    .stMarkdown p { color: #cbd5e1 !important; }
    .disclaimer { border: 1px solid rgba(255,255,255,0.06); padding: 8px; border-radius: 6px; color: #9aa8bf; background-color: rgba(255,255,255,0.02); }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
    st.title('🌙 SoulScript')
    st.markdown('<div class="disclaimer">A quiet space to write and understand yourself.</div>', unsafe_allow_html=True)

    clean_fn = import_clean_text()
    vec, clf, le = load_artifacts()

    st.header('Write a short journal entry')

    # Daily prompt (optional inspiration)
    prompts = [
        "What's been taking up space in your mind today?",
        "Describe one moment from today, big or small.",
        "What do you wish someone understood about how you're feeling?",
        "What's one thing you're grateful for right now?",
        "Name a small success you had today, however minor.",
        "What emotion has shown up most for you today?",
        "If you could say one thing to your future self, what would it be?",
        "What would make today feel a little easier?",
    ]
    if 'prompt_idx' not in st.session_state:
        st.session_state['prompt_idx'] = random.randrange(len(prompts))
    prompt_col1, prompt_col2 = st.columns([8,1])
    with prompt_col1:
        st.markdown(f"**Prompt:** {prompts[st.session_state['prompt_idx']]}")
    with prompt_col2:
        if st.button('🔄 New prompt'):
            st.session_state['prompt_idx'] = random.randrange(len(prompts))

    # Self-reported wellness inputs (placed above the textarea)
    cols = st.columns([1, 1, 1])
    with cols[0]:
        sleep_choice = st.selectbox("Sleep (hours)", options=["Less than 4 hours", "4-6 hours", "6-8 hours", "8+ hours"], index=2)
        # map selection to approximate numeric hours for scoring
        _sleep_map = {
            "Less than 4 hours": 3.0,
            "4-6 hours": 5.0,
            "6-8 hours": 7.0,
            "8+ hours": 9.0,
        }
        sleep_hours = _sleep_map.get(sleep_choice, 7.0)
    with cols[1]:
        # use descriptive stress options and map to a calmness numeric (0-10)
        stress_choice = st.select_slider("How stressed do you feel right now?",
                                         options=["Very calm", "Mostly calm", "A bit tense", "Stressed", "Very overwhelmed"],
                                         value="A bit tense")
        _stress_map = {
            "Very calm": 10.0,
            "Mostly calm": 7.5,
            "A bit tense": 5.0,
            "Stressed": 2.5,
            "Very overwhelmed": 0.0,
        }
        stress_level = _stress_map.get(stress_choice, 5.0)
    with cols[2]:
        mood_choice = st.select_slider("How's your mood right now?",
                                       options=['😊 Feeling good', '🙂 Doing okay', '😐 Just neutral', '😔 Having a hard time', '😰 Really struggling'],
                                       value='😐 Just neutral')

    entry = st.text_area('Your entry', height=160)
    submit = st.button('Reflect')

    if 'entries' not in st.session_state:
        st.session_state['entries'] = []
    if 'revealed' not in st.session_state:
        st.session_state['revealed'] = {}

    if submit and entry and entry.strip():
        ts = time.time()
        pred_label, probs, top_features = predict_and_explain(entry, vec, clf, le, clean_fn)
        # save session
        snippet = entry.strip()[:240]
        confidence = float(np.max(probs))

        # compute combined wellness score
        wellness, subs = combine_wellness_score(pred_label, confidence, sleep_hours, stress_level, mood_choice)

        # Critical safety helpline card: show immediately if suicidal or very low wellness
        if (pred_label == 'Suicidal') or (wellness < 20):
            helpline_html = """
            <div style='background-color:#2b0b0b; padding:12px; border-radius:8px; border:2px solid #c62828; color:#fff;'>
              <strong>If you're having thoughts of suicide, please reach out right now — you deserve support.</strong>
              <div style='margin-top:8px;'>
                🇮🇳 TeleMANAS (India, Govt., 24/7, free): 14416 or 1-800-891-4416<br/>
                🇮🇳 Vandrevala Foundation: +91-9999-666-555<br/>
                If you're outside India, visit findahelpline.com for helplines in your country.
              </div>
            </div>
            """
            st.markdown(helpline_html, unsafe_allow_html=True)

        st.session_state['entries'].append({
            'ts': ts,
            'text': snippet,
            'pred': pred_label,
            'conf': confidence,
            'probs': probs.tolist(),
            'sleep': float(sleep_hours),
            'stress': float(stress_level),
            'mood': mood_choice,
            'wellness': wellness,
            'subscores': subs,
        })

        # Output
        st.subheader('Reflection')
        st.write('Predicted pattern:', pred_label)
        # Wellness metric (delta vs previous entry)
        prev_wellness = None
        if len(st.session_state['entries']) > 1:
            prev_wellness = st.session_state['entries'][-2].get('wellness')
        delta = None
        if prev_wellness is not None:
            delta = wellness - float(prev_wellness)
        st.metric('Wellness Score', f"{wellness:.0f}", delta=f"{delta:+.0f}" if delta is not None else None)

        # Log subscores for debugging to terminal
        print(f"SUBSCORES: {subs}")

        # Suggestions card
        suggestions = get_suggestions(wellness, subs)
        sugg_html = """
        <div style='background-color:#111227; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);'>
          <h4 style='margin:0 0 6px 0; color:#e6eef8;'>💡 Gentle Suggestions</h4>
          <ul style='margin:0 0 0 18px; color:#cbd5e1;'>
        """
        for s in suggestions:
            sugg_html += f"<li>{s}</li>"
        sugg_html += "</ul></div>"
        st.markdown(sugg_html, unsafe_allow_html=True)
        # Low-confidence UX note
        if confidence < 0.4:
            st.info("This entry didn't show strong signals — try adding more detail for a clearer reflection.")
        st.write('This is a reflection tool, not a medical diagnosis. If you\'re struggling, please reach out to a mental health professional.')

        # Show bar chart of probabilities using Plotly for styling
        prob_df = pd.DataFrame({'class': list(le.classes_), 'prob': probs})
        prob_df = prob_df.sort_values('prob', ascending=True)
        fig = go.Figure(go.Bar(x=prob_df['prob'], y=prob_df['class'], orientation='h', marker_color='#6a5acd'))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, color='#cbd5e1'), yaxis=dict(color='#cbd5e1'))
        st.plotly_chart(fig, use_container_width=True)

        # Explainability: hide by default in a collapsed expander for non-technical users
        with st.expander("🔍 Technical details (for those curious)", expanded=False):
            st.subheader('Words that influenced this reflection')
            if top_features:
                tf_df = pd.DataFrame(top_features, columns=['feature', 'contribution'])
                st.table(tf_df)
            else:
                st.write('No top features identified for this entry.')

    # Session trends
    st.write('---')
    st.header('Session history')
    entries = st.session_state.get('entries', [])
    if entries:
        df = pd.DataFrame(entries)
        df['idx'] = range(1, len(df) + 1)

        # Color palette (muted, dark-theme friendly)
        color_map = {
            'Normal': '#3fc1c9',        # soft teal
            'Stress': '#f6a00a',        # amber
            'Anxiety': '#ff6b6b',       # orange-red
            'Bipolar': '#9b59b6',       # purple
            'Depression': '#1f4b77',    # deep blue
            'Personality disorder': '#6d5aa3',
            'Suicidal': '#7a1130',      # muted deep red
        }

        # Use combined wellness score (0-100) for the y-axis
        df['y'] = df['wellness'].astype(float)
        df['color'] = df['pred'].map(color_map)

        # handle edge case: less than 2 points => show placeholder
        if len(df) < 2:
            st.info('Your journey will appear here as you add reflections')
        else:
            # build hover text (include wellness)
            hover_texts = [f"#{row['idx']} — {row['pred']}<br>{row['text'][:140]}<br>Confidence: {row['conf']:.2f}<br>Wellness: {row['wellness']:.0f}" for _, row in df.iterrows()]

            # Create flowing wave using Plotly spline with layered glow, fill, and colored markers
            fig = go.Figure()

            # prepare customdata: [pred, confidence_percent, snippet, wellness]
            customdata = list(zip(df['pred'].tolist(), (df['conf'] * 100).tolist(), df['text'].str.slice(0, 40).tolist(), df['wellness'].tolist()))

            # glow layer (thick, semi-transparent)
            fig.add_trace(go.Scatter(
                x=df['idx'],
                y=df['y'],
                mode='lines',
                line=dict(shape='spline', color='rgba(147,112,219,0.18)', width=8),
                hoverinfo='skip',
                showlegend=False,
            ))

            # filled area under curve (soft purple glow)
            fig.add_trace(go.Scatter(
                x=df['idx'],
                y=df['y'],
                mode='lines',
                line=dict(shape='spline', color='rgba(0,0,0,0)', width=0),
                fill='tozeroy',
                fillcolor='rgba(147,112,219,0.18)',
                hoverinfo='skip',
                showlegend=False,
            ))

            # main line with markers
            fig.add_trace(go.Scatter(
                x=df['idx'],
                y=df['y'],
                mode='lines+markers',
                line=dict(shape='spline', color='rgba(220,220,255,1)', width=3),
                marker=dict(size=13, color=df['color'], line=dict(width=2, color='rgba(255,255,255,0.3)')),
                customdata=customdata,
                hovertemplate="Entry #%{x}<br>Mood: %{customdata[0]}<br>Confidence: %{customdata[1]:.0f}%<br>Wellness: %{customdata[3]:.0f}<br>%{customdata[2]}<extra></extra>",
                showlegend=False,
            ))

            # add invisible traces per category to build a legend with consistent colors
            for cat, col in color_map.items():
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, color=col, line=dict(width=2, color='rgba(255,255,255,0.3)')), name=cat))

            # y-axis is 0..100 wellness score now
            fig.update_layout(
                title=dict(text='Your Emotional Journey (combined wellness)', x=0.02, xanchor='left', font=dict(color='#e6eef8')),
                xaxis=dict(title='Entry #', color='#cbd5e1', showgrid=False),
                yaxis=dict(title='Wellness (0-100)', range=[0, 100], color='#cbd5e1', gridcolor='rgba(255,255,255,0.03)', zeroline=False),
                plot_bgcolor='#0f0f1e',
                paper_bgcolor='#0f0f1e',
                margin=dict(l=40, r=24, t=60, b=40),
                height=480,
            )

            st.plotly_chart(fig, use_container_width=True, height=480)

        # privacy note and session stats
        st.markdown("🔒 Entries are private by default — click 👁️ to view any reflection.")
        # Session streak and total words
        n_entries = len(df)
        total_words = sum(len(e.get('text','').split()) for e in entries)
        stats_col1, stats_col2 = st.columns([1,1])
        with stats_col1:
            st.markdown(f"✍️ {n_entries} reflections this session")
        with stats_col2:
            st.markdown(f"📝 {total_words} words written")

        # Download/export
        include_text = st.checkbox('Include full journal text in export?', value=False)
        export_lines = []
        for e in entries:
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['ts']))
            if include_text:
                export_lines.append(f"{ts_str} | {e['pred']} | Wellness: {e['wellness']:.1f}\n{e['text']}\n\n")
            else:
                export_lines.append(f"{ts_str} | {e['pred']} | Wellness: {e['wellness']:.1f}\n")
        export_content = '\n'.join(export_lines)
        st.download_button('Download my reflections', export_content, file_name='soulscript_session.txt')

        # Build masked session display with per-entry reveal
        for i, e in enumerate(entries, start=1):
            row_key = f"reveal_{i}"
            revealed = st.session_state['revealed'].get(row_key, False)
            col1, col2, col3 = st.columns([6,2,1])
            with col1:
                ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['ts']))
                st.markdown(f"**#{i} — {ts_str}**  ")
                if revealed:
                    st.markdown(f"> {e['text']} ")
                else:
                    st.markdown(
                        "> 🔒 [entry hidden — click 👁️ to reveal]"
                    )
            with col2:
                st.markdown(f"**Mood:** {e['pred']}  \n**Wellness:** {e['wellness']:.0f}")
            with col3:
                if st.button('👁️ Reveal', key=row_key):
                    st.session_state['revealed'][row_key] = not revealed
                    st.rerun()
    else:
        st.write('No entries this session yet.')


if __name__ == '__main__':
    main()
