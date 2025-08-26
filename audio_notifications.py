import streamlit as st
import base64
import time
from datetime import datetime, timedelta

class AudioNotifications:
    """Handle audio notifications for trading signals"""
    
    def __init__(self):
        self.notification_history = []
        self.cooldown_seconds = 10  # Minimum 10 seconds between notifications
        
    def generate_beep_audio(self, frequency=800, duration=0.5, sample_rate=22050):
        """Generate a simple beep sound"""
        import numpy as np
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio = (wave * 32767).astype(np.int16)
        return audio
    
    def create_audio_html(self, audio_data, sample_rate=22050, autoplay=True):
        """Create HTML5 audio element with base64 encoded audio"""
        try:
            import io
            import wave
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Get WAV data and encode to base64
            wav_data = buffer.getvalue()
            b64_audio = base64.b64encode(wav_data).decode()
            
            # Create HTML audio element
            audio_html = f'''
            <audio id="notification-sound" {"autoplay" if autoplay else ""}>
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            '''
            
            return audio_html
            
        except Exception as e:
            st.error(f"Error creating audio: {e}")
            return None
    
    def create_text_to_speech_html(self, text, autoplay=True):
        """Create HTML with JavaScript text-to-speech"""
        speech_html = f'''
        <script>
            function speakText() {{
                if ('speechSynthesis' in window) {{
                    var utterance = new SpeechSynthesisUtterance('{text}');
                    utterance.rate = 1.2;
                    utterance.pitch = 1.0;
                    utterance.volume = 0.8;
                    
                    // Set voice to a clear English voice if available
                    var voices = speechSynthesis.getVoices();
                    var englishVoice = voices.find(voice => voice.lang.startsWith('en'));
                    if (englishVoice) {{
                        utterance.voice = englishVoice;
                    }}
                    
                    speechSynthesis.speak(utterance);
                }}
            }}
            
            {'speakText();' if autoplay else ''}
        </script>
        '''
        return speech_html
    
    def should_play_notification(self):
        """Check if enough time has passed since last notification"""
        current_time = datetime.now()
        
        if not self.notification_history:
            return True
        
        last_notification = self.notification_history[-1]
        time_since_last = (current_time - last_notification).total_seconds()
        
        return time_since_last >= self.cooldown_seconds
    
    def play_signal_notification(self, signal, notification_type='beep'):
        """Play notification for new trading signal"""
        try:
            if not self.should_play_notification():
                return
            
            # Record notification time
            self.notification_history.append(datetime.now())
            
            # Keep only last 10 notifications in history
            if len(self.notification_history) > 10:
                self.notification_history = self.notification_history[-10:]
            
            symbol = signal['symbol'].replace('.NS', '').replace('-USD', '')
            action = signal['action']
            confidence = signal.get('confidence', 0)
            
            if notification_type == 'voice':
                # Text-to-speech notification
                text = f"New {action} signal for {symbol} with {confidence:.0%} confidence"
                audio_html = self.create_text_to_speech_html(text)
                
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
            elif notification_type == 'beep':
                # Simple beep notification
                try:
                    import numpy as np
                    
                    # Different tones for buy/sell
                    frequency = 900 if action == 'BUY' else 600
                    duration = 0.6 if confidence > 0.9 else 0.4
                    
                    audio_data = self.generate_beep_audio(frequency, duration)
                    audio_html = self.create_audio_html(audio_data)
                    
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                except ImportError:
                    # Fallback to browser notification
                    st.success(f"🔔 New {action} Signal: {symbol} ({confidence:.0%})")
                    
            elif notification_type == 'chime':
                # Multiple tone chime
                try:
                    import numpy as np
                    
                    # Create ascending chime for buy, descending for sell
                    frequencies = [600, 800, 1000] if action == 'BUY' else [1000, 800, 600]
                    
                    combined_audio = np.array([])
                    for freq in frequencies:
                        tone = self.generate_beep_audio(freq, 0.2)
                        combined_audio = np.concatenate([combined_audio, tone])
                    
                    audio_html = self.create_audio_html(combined_audio)
                    
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                except ImportError:
                    st.success(f"🎵 New {action} Signal: {symbol} ({confidence:.0%})")
            
            # Also show visual notification
            self.show_visual_notification(signal)
            
        except Exception as e:
            st.error(f"Notification error: {e}")
    
    def show_visual_notification(self, signal):
        """Show visual notification alongside audio"""
        symbol = signal['symbol'].replace('.NS', '').replace('-USD', '')
        action = signal['action']
        confidence = signal.get('confidence', 0)
        strategy = signal.get('strategy', 'Unknown')
        
        # Create a temporary visual alert
        if action == 'BUY':
            st.success(f"📢 **NEW BUY SIGNAL** - {symbol} | {confidence:.0%} confidence | {strategy}")
        else:
            st.error(f"📢 **NEW SELL SIGNAL** - {symbol} | {confidence:.0%} confidence | {strategy}")
    
    def create_notification_controls(self):
        """Create UI controls for notification settings"""
        st.sidebar.markdown("---")
        st.sidebar.header("🔔 Audio Notifications")
        
        # Enable/disable notifications
        audio_enabled = st.sidebar.checkbox(
            "Enable Signal Notifications", 
            value=st.session_state.get('audio_notifications_enabled', False),
            help="Play sound when new trading signals are generated"
        )
        
        if audio_enabled:
            # Notification type selection
            notification_types = {
                'beep': '🔈 Simple Beep',
                'voice': '🗣️ Voice Alert', 
                'chime': '🎵 Chime'
            }
            
            notification_type = st.sidebar.selectbox(
                "Notification Type",
                options=list(notification_types.keys()),
                format_func=lambda x: notification_types[x],
                index=0
            )
            
            # Volume control (visual indicator)
            volume = st.sidebar.slider(
                "Alert Sensitivity",
                min_value=1,
                max_value=3,
                value=2,
                help="1=High confidence only, 2=Medium+, 3=All signals"
            )
            
            # Cooldown setting
            cooldown = st.sidebar.slider(
                "Cooldown (seconds)",
                min_value=5,
                max_value=60,
                value=10,
                help="Minimum time between notifications"
            )
            
            self.cooldown_seconds = cooldown
            
            # Test notification button
            if st.sidebar.button("🔊 Test Notification"):
                test_signal = {
                    'symbol': 'BTC-USD',
                    'action': 'BUY',
                    'confidence': 0.95,
                    'strategy': 'Test Signal',
                    'timestamp': datetime.now()
                }
                self.play_signal_notification(test_signal, notification_type)
                st.sidebar.success("Test notification played!")
            
            # Store settings in session state
            st.session_state.audio_notifications_enabled = True
            st.session_state.notification_type = notification_type
            st.session_state.notification_volume = volume
            
            # Show current status
            st.sidebar.success("🟢 Notifications Active")
            
        else:
            st.session_state.audio_notifications_enabled = False
            st.sidebar.info("🔕 Notifications Disabled")
        
        return audio_enabled
    
    def check_for_new_signals(self, current_signals, previous_signals):
        """Check if there are new signals and trigger notifications"""
        if not st.session_state.get('audio_notifications_enabled', False):
            return
        
        # Find new signals by comparing with previous signals
        new_signals = []
        
        if not previous_signals:
            # First load - don't notify for existing signals
            return
        
        for signal in current_signals:
            # Check if this is a genuinely new signal
            is_new = True
            for prev_signal in previous_signals:
                if (signal.get('symbol') == prev_signal.get('symbol') and
                    signal.get('action') == prev_signal.get('action') and
                    signal.get('timestamp') == prev_signal.get('timestamp')):
                    is_new = False
                    break
            
            if is_new:
                new_signals.append(signal)
        
        # Play notifications for new signals
        notification_type = st.session_state.get('notification_type', 'beep')
        volume_threshold = st.session_state.get('notification_volume', 2)
        
        for signal in new_signals:
            confidence = signal.get('confidence', 0)
            
            # Filter by volume/sensitivity setting
            should_notify = False
            if volume_threshold == 1 and confidence >= 0.95:  # High confidence only
                should_notify = True
            elif volume_threshold == 2 and confidence >= 0.85:  # Medium+ confidence
                should_notify = True
            elif volume_threshold == 3 and confidence >= 0.60:  # All signals
                should_notify = True
            
            if should_notify:
                self.play_signal_notification(signal, notification_type)
        
        return len(new_signals)