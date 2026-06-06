"""
AI聊天系统 - 样式定义模块

职责：
- 定义页面样式相关的代码，包括 CSS 字符串、样式配置、主题色、字体、布局的静态定义
- 提供可复用的样式变量和类名定义
- 支持明暗主题切换
"""

# 主题配置 - 包含所有颜色、间距、阴影等设计变量
THEME_CONFIG = {
    # 主色调
    'primary': '#3B82F6',
    'primary_light': '#589CFC',
    'secondary': '#8B5CF6',
    
    # 文字颜色
    'text_primary': '#1F2937',
    'text_secondary': '#6B7280',
    'text-muted': '#9CA3AF',
    
    # 边框和背景
    'border': '#E5E7EB',
    'border_light': '#F3F4F6',
    'bg': '#F9FAFB',
    'panel': '#FFFFFF',
    'bg_chat': '#F3F4F6',
    
    # 状态颜色
    'danger': '#EF4444',
    'success': '#10B981',
    
    # 阴影效果
    'shadow_sm': '0 1px 2px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)',
    'shadow_md': '0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02)',
    'shadow_lg': '0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03)',
    'shadow_hover': '0 12px 20px rgba(0, 0, 0, 0.08), 0 6px 8px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.03)',
    
    # 圆角
    'radius_sm': '12px',
    'radius_md': '16px',
    'radius_lg': '20px',
    'radius_xl': '24px',
    'radius_bubble': '18px',
    
    # 过渡动画
    'transition_fast': '0.15s ease-in-out',
    'transition_normal': '0.2s ease-in-out',
    'transition_slow': '0.3s ease-in-out',
}

# 完整的 CSS 样式字符串
CSS_STYLES = """
:root{
  --primary: #3B82F6;
  --primary-light: #589CFC;
  --secondary: #8B5CF6;
  --text-primary: #1F2937;
  --text-secondary: #6B7280;
  --text-muted: #9CA3AF;
  --border: #E5E7EB;
  --border-light: #F3F4F6;
  --bg: #F9FAFB;
  --panel: #FFFFFF;
  --bg-chat: #F3F4F6;
  --danger: #EF4444;
  --success: #10B981;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.05), 0 4px 6px rgba(0, 0, 0, 0.04), 0 2px 4px rgba(0, 0, 0, 0.03);
  --shadow-hover: 0 12px 20px rgba(0, 0, 0, 0.08), 0 6px 8px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.03);
  --radius-sm: 12px;
  --radius-md: 16px;
  --radius-lg: 20px;
  --radius-xl: 24px;
  --radius-bubble: 18px;
  --transition-fast: 0.15s ease-in-out;
  --transition-normal: 0.2s ease-in-out;
  --transition-slow: 0.3s ease-in-out;
}

@media (prefers-color-scheme: dark) {
  :root{
    --primary: #60A5FA;
    --primary-light: #93C5FD;
    --secondary: #A78BFA;
    --text-primary: #F9FAFB;
    --text-secondary: #D1D5DB;
    --text-muted: #9CA3AF;
    --border: #374151;
    --border-light: #1F2937;
    --bg: #111827;
    --panel: #1F2937;
    --bg-chat: #111827;
    --danger: #F87171;
    --success: #34D399;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2), 0 1px 3px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.35), 0 4px 6px rgba(0, 0, 0, 0.25), 0 2px 4px rgba(0, 0, 0, 0.15);
    --shadow-hover: 0 12px 20px rgba(0, 0, 0, 0.4), 0 6px 8px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  body{
    background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #1f2937 100%);
  }
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body{
  margin:0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background: linear-gradient(135deg, #F3F4F6 0%, #E0E7FF 50%, #F3F4F6 100%);
  color: var(--text-primary);
  font-weight: 400;
  min-height: 100vh;
  transition: background 0.3s ease-in-out;
}
a{ color: inherit; }
.wrap{ max-width: 1300px; margin: 0 auto; padding: 20px 24px; }
.topbar{
  display:flex; align-items:center; justify-content:space-between;
  gap:16px; padding: 14px 20px; border: 1px solid var(--border);
  background: var(--panel);
  border-radius: var(--radius-lg); box-shadow: var(--shadow-md);
}
.brand{ display:flex; align-items:center; gap:12px; }
.logo{
  width: 42px; height: 42px; border-radius: var(--radius-sm);
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  display:grid; place-items:center; font-weight: 700;
  color: white;
  font-size: 18px;
  box-shadow: var(--shadow-sm);
}
.title{ font-weight: 600; letter-spacing: -.02em; font-size: 18px; color: var(--text-primary); }
.sub{ font-size: 12px; color: var(--text-secondary); margin-top: 1px; }
.row{ 
  display:flex; 
  gap: 20px; 
  margin-top: 20px; 
  position: relative;
  height: calc(100vh - 140px);
}
.left{
  width: 320px; 
  flex: 0 0 320px;
  border: 1px solid var(--border); 
  border-radius: var(--radius-lg);
  background: var(--panel); 
  box-shadow: var(--shadow-md);
  padding: 0;
  overflow: hidden;
  position: fixed;
  left: calc(50% - 630px);
  top: 100px;
  height: calc(100vh - 140px);
  overflow-y: auto;
  z-index: 100;
}
.left.hidden{
  display: none;
}
.main{
  flex: 1 1 auto;
  margin-left: 340px;
  border: 1px solid var(--border); 
  border-radius: var(--radius-lg);
  background: var(--panel); 
  box-shadow: var(--shadow-md);
  padding: 0;
  height: calc(100vh - 140px);
  display:flex; 
  flex-direction:column;
  overflow: hidden;
}
.card{
  background: var(--panel);
  padding: 16px;
}
.card-section{
  padding: 16px;
  border-bottom: 1px solid var(--border-light);
}
.card-section:last-child{
  border-bottom: none;
}
.card-section.compact{
  padding: 14px 16px;
}
.card-header{
  background: linear-gradient(135deg, var(--bg-chat), var(--border-light));
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-light);
}
.kpis{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 8px; }
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding: 5px 12px; border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-chat);
  font-size: 12px;
  color: var(--text-secondary);
  transition: all var(--transition-fast);
}
.chip.success{
  background: rgba(16, 185, 129, 0.1);
  border-color: rgba(16, 185, 129, 0.3);
  color: var(--success);
}
.grid2{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
label{ font-size: 12px; color: var(--text-secondary); display:block; margin-bottom: 6px; font-weight: 500; }
.form-label{ font-size: 13px; color: var(--text-primary); display:block; margin-bottom: 8px; font-weight: 500; }
input, select, textarea{
  width: 100%;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-chat);
  color: var(--text-primary);
  padding: 10px 14px;
  outline: none;
  transition: border-color var(--transition-fast), background var(--transition-fast), box-shadow var(--transition-fast);
  font-size: 14px;
}
input:hover, select:hover, textarea:hover{
  border-color: var(--text-muted);
}
input:focus, select:focus, textarea:focus{
  border-color: var(--primary);
  background: var(--panel);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
textarea{ min-height: 80px; resize: vertical; }
input::placeholder, textarea::placeholder{ color: var(--text-muted); }

.btn{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-chat);
  color: var(--text-primary);
  padding: 10px 16px;
  cursor: pointer;
  transition: all var(--transition-normal);
  font-weight: 500;
  font-size: 13px;
  line-height: 1;
}
.btn:hover{ 
  background: var(--border-light); 
  border-color: var(--text-muted);
  transform: translateY(-1px);
}
.btn:active{ 
  transform: translateY(0); 
}
.btn:disabled{
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}
.btn.primary{ 
  background: linear-gradient(135deg, var(--primary), var(--primary-light)); 
  border-color: var(--primary); 
  color: white;
  box-shadow: var(--shadow-sm);
}
.btn.primary:hover{ 
  background: linear-gradient(135deg, var(--primary-light), var(--primary)); 
  border-color: var(--primary-light);
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.btn.primary:active{ 
  transform: translateY(-1px); 
}
.btn.good{ 
  background: linear-gradient(135deg, var(--success), #34D399); 
  border-color: var(--success); 
  color: white;
  box-shadow: var(--shadow-sm);
}
.btn.good:hover{ 
  background: linear-gradient(135deg, #34D399, var(--success)); 
  border-color: #34D399;
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.btn.danger{ 
  background: transparent; 
  border-color: var(--danger); 
  color: var(--danger);
}
.btn.danger:hover{ 
  background: rgba(239, 68, 68, 0.05); 
}
.btn.outline{
  background: transparent;
  border-color: var(--border);
  color: var(--text-secondary);
}
.btn.outline:hover{
  background: var(--bg-chat);
  border-color: var(--text-muted);
}
.btn.small{
  padding: 6px 12px;
  font-size: 12px;
}
.btn.full{ width: 100%; }
.text-muted{ color: var(--text-muted); font-size: 12px; }
.text-secondary{ color: var(--text-secondary); font-size: 13px; }
.hr{ height:1px; background: var(--border-light); margin: 0; }
.hidden{ display:none !important; }
.roleList{
  display:flex;
  flex-direction:column;
  gap: 8px;
  max-height: 380px;
  overflow:auto;
  padding-right: 4px;
}
.roleList::-webkit-scrollbar{
  width: 6px;
}
.roleList::-webkit-scrollbar-track{
  background: transparent;
}
.roleList::-webkit-scrollbar-thumb{
  background: var(--border);
  border-radius: 3px;
}
.roleList::-webkit-scrollbar-thumb:hover{
  background: var(--text-muted);
}
.roleCard{
  display:flex;
  align-items:flex-start;
  gap:12px;
  padding:14px;
  border:1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--panel);
  cursor:pointer;
  transition: all var(--transition-normal);
  position: relative;
}
.roleCard:hover{
  border-color: var(--primary);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}
.roleCard.active{
  border-color: var(--primary);
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.06));
  box-shadow: var(--shadow-lg);
}
.roleCard.active::before{
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 24px;
  background: linear-gradient(180deg, var(--primary), var(--secondary));
  border-radius: 0 3px 3px 0;
}
.roleIcon{
  width:44px;
  height:44px;
  border-radius: var(--radius-sm);
  flex:0 0 44px;
  display:grid;
  place-items:center;
  font-size: 16px;
  font-weight: 700;
  color:#fff;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  box-shadow: var(--shadow-sm);
}
.roleMeta{
  min-width:0;
  flex:1;
}
.roleName{
  font-size:15px;
  font-weight:600;
  color:var(--text-primary);
  margin-bottom:3px;
}
.roleBrief{
  font-size:12px;
  color:var(--text-secondary);
  line-height:1.5;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  overflow:hidden;
}

.chatHeader{ 
  display:flex; 
  align-items:flex-start; 
  justify-content:space-between; 
  gap: 12px; 
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-light);
}
.chatTitle{ font-size: 16px; font-weight: 600; color: var(--text-primary); }
.chatDesc{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }
#chatScreen{
  display: flex;
  flex-direction: column;
  height: 100%;
}
.chatWrap{
  flex: 1 1 auto;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 16px 20px;
  -webkit-overflow-scrolling: touch;
  background: var(--bg-chat);
}
.chatWrap::-webkit-scrollbar{
  width: 6px;
}
.chatWrap::-webkit-scrollbar-track{
  background: transparent;
}
.chatWrap::-webkit-scrollbar-thumb{
  background: var(--border);
  border-radius: 3px;
}
.chatWrap::-webkit-scrollbar-thumb:hover{
  background: var(--text-muted);
}
.msg{ 
  display:flex; 
  gap: 12px; 
  margin: 16px 0; 
  max-width: 100%;
}
.msg.user{ justify-content: flex-start; flex-direction: row-reverse; }

.msg-image{
  max-width: 200px;
  max-height: 200px;
  border-radius: var(--radius-sm);
  object-fit: cover;
  margin-bottom: 8px;
}

.file-preview{
  position: absolute;
  bottom: 100%;
  left: 0;
  right: 0;
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  margin-bottom: 8px;
  display: none;
  flex-direction: column;
  gap: 8px;
}

.file-preview.show{
  display: flex;
}

.preview-image{
  max-width: 150px;
  max-height: 150px;
  border-radius: var(--radius-sm);
  object-fit: cover;
  align-self: flex-start;
}

.preview-info{
  display: flex;
  align-items: center;
  gap: 10px;
}

.preview-name{
  font-size: 13px;
  color: var(--text-primary);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.btn-remove-file{
  padding: 4px 10px;
  font-size: 12px;
  background: transparent;
  border: 1px solid var(--danger);
  color: var(--danger);
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.btn-remove-file:hover{
  background: rgba(239, 68, 68, 0.08);
}

.recording-indicator{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-sm);
  color: var(--danger);
  font-size: 13px;
  animation: pulse 1.5s ease-in-out infinite;
}

.recording-dot{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--danger);
  animation: blink 1s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}

.recording-text{
  font-weight: 500;
}

.msg.assistant{ justify-content: flex-start; }
.avatar{
  width: 36px; 
  height: 36px; 
  border-radius: 50%;
  display:grid; 
  place-items:center; 
  font-weight: 600;
  background: var(--border-light);
  border: 1px solid var(--border);
  flex: 0 0 36px;
  color: var(--text-secondary);
  font-size: 14px;
}
.bubble{
  border-radius: var(--radius-bubble);
  padding: 14px 18px;
  max-width: 75%;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
  font-size: 15px;
  position: relative;
}
.msg.user .avatar{ 
  background: linear-gradient(135deg, var(--secondary), #F472B6); 
  border-color: rgba(139, 92, 246, 0.3); 
  color: white;
  box-shadow: var(--shadow-sm);
}
.msg.user .bubble{ 
  background: linear-gradient(135deg, var(--primary-light), var(--primary)); 
  color: white;
  border-radius: var(--radius-bubble) var(--radius-bubble) 6px var(--radius-bubble);
  box-shadow: var(--shadow-sm);
}
.msg.user .bubble::after{
  content: '';
  position: absolute;
  right: -6px;
  top: 12px;
  width: 0;
  height: 0;
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-left: 6px solid var(--primary);
}
.msg.assistant .avatar{ 
  background: linear-gradient(135deg, var(--primary), var(--secondary)); 
  border-color: rgba(59, 130, 246, 0.3); 
  color: white;
  box-shadow: var(--shadow-sm);
}
.msg.assistant .bubble{ 
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--text-primary);
  border-radius: var(--radius-bubble) var(--radius-bubble) var(--radius-bubble) 6px;
  box-shadow: var(--shadow-sm);
}
.msg.assistant .bubble::after{
  content: '';
  position: absolute;
  left: -6px;
  top: 12px;
  width: 0;
  height: 0;
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-right: 6px solid var(--border);
}
.msg-time{
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  text-align: right;
}
.msg.user .msg-time{
  text-align: left;
}

.composer{
  padding: 16px 20px;
  background: var(--panel);
  border-top: 1px solid var(--border-light);
  display:flex; 
  gap: 12px; 
  align-items:flex-end;
  position: relative;
}
.composer .input-wrapper{
  flex: 1;
  position: relative;
}
.composer textarea{ 
  width: 100%;
  min-height: 56px; 
  max-height: 180px;
  border-radius: var(--radius-xl);
  border: 1px solid var(--border);
  background: var(--bg-chat);
  padding: 14px 50px 14px 16px;
  resize: none;
  transition: all var(--transition-normal);
  font-size: 15px;
  line-height: 1.5;
}
.composer textarea:focus{
  border-color: var(--primary);
  background: var(--panel);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.08), 0 4px 12px rgba(59, 130, 246, 0.08);
}
.composer-actions{
  position: absolute;
  right: 12px;
  bottom: 12px;
  display: flex;
  gap: 8px;
}
.composer-btn{
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: all var(--transition-fast);
}
.composer-btn:hover{
  background: var(--border-light);
  color: var(--text-primary);
}
.composer-hint{
  position: absolute;
  right: 12px;
  top: 8px;
  font-size: 11px;
  color: var(--text-muted);
}
.emoji-panel{
  position: absolute;
  top: calc(100% + 8px);
  left: 20px;
  right: 20px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  padding: 12px;
  z-index: 200;
  max-height: 300px;
  overflow-y: auto;
}
.emoji-header{
  display: flex;
  align-items: center;
  padding: 0 8px 12px;
  border-bottom: 1px solid var(--border-light);
  margin-bottom: 12px;
}
.emoji-grid{
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 4px;
  max-height: 200px;
  overflow-y: auto;
}
.emoji-item{
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  cursor: pointer;
  border-radius: 8px;
  transition: all var(--transition-fast);
}
.emoji-item:hover{
  background: var(--border-light);
  transform: scale(1.15);
}

.quick-prompts{
  padding: 12px 20px;
  background: var(--panel);
  border-top: 1px solid var(--border-light);
  animation: slideUp 0.2s ease-out;
}
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
.quick-prompts-header{
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-light);
}
.quick-prompts-list{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.quick-prompt{
  padding: 8px 14px;
  background: var(--bg-chat);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.quick-prompt:hover{
  background: rgba(59, 130, 246, 0.08);
  border-color: var(--primary);
  color: var(--primary);
}

.captcha-row{
  display:flex; gap: 10px; align-items:center;
}
.captcha-img{
  width: 100px;
  height: 40px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--gray-50);
  cursor: pointer;
}

.loading-spinner{
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--gray-200);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 980px){
  .row{ flex-direction:column; }
  .left{ width: 100%; flex: 1 1 auto; }
}
"""
