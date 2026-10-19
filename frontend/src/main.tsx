import React, { FormEvent, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Room, RoomEvent, Track } from 'livekit-client';
import { AudioLines, ArrowUp, ArrowUpRight, BookOpen, Check, CheckCheck, ChevronDown,
  ChevronRight, CircleHelp, Clock3, FileText, Headphones, LoaderCircle, MessageSquare,
  Mic, MicOff, Phone, PhoneOff, Plus, Radio, Search, Settings2, ShieldCheck, Sparkles,
  Upload, X, Zap, Activity, Cable, Cpu, ExternalLink, Pencil, Trash2, Save, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import './index.css';
import './extras.css';

type Citation = { chunk_id: string; document_id: string; document_name: string; heading: string; text: string; score: number; version: string };
type Message = { id: string; session_id: string; turn_id: string; role: 'user' | 'assistant'; text: string; citations: Citation[]; metrics: Record<string, any>; interrupted: boolean; answer_mode: string; created_at: string };
type KnowledgeProfile = { role: string; scope: string; context_fields: string; clarification_policy: string; response_style: string; aliases: string };
type Knowledge = { id: string; name: string; profile: KnowledgeProfile; document_count?: number; updated_at: string };
type Session = { id: string; title: string; backend: string; status: string; created_at: string; updated_at: string; messages?: Message[]; knowledge_base_id: string; knowledge_profile: Partial<Knowledge> };
type Status = { default_backend: string; livekit_ready: boolean; agent_ready: boolean; voice_ready: boolean; llm_configured: boolean; llm_model: string; stt_provider: string; tts_provider: string; stt_model: string; tts_model: string; missing_resources: string[]; document_count: number; chunk_count: number; backends: { id: string; name: string; ready: boolean }[]; vector_ready: boolean; embedding_model: string; embedding_dimension: number; embedding_provider: string; deployment: string; livekit_public_url: string; vector_dashboard_url: string; knowledge_error: string };
type Doc = { id: string; filename: string; title: string; version: string; chunks: number; characters: number; content?: string; index_status: string; knowledge_base_id: string };
type Page = 'chat' | 'knowledge' | 'metrics';

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const r = await fetch('/api' + path, { credentials: 'same-origin', ...options,
    headers: { ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options?.headers } });
  if (r.status === 401) { window.location.assign('/login'); throw new Error('请先登录。'); }
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请重试。');
  return data;
}
const post = <T,>(path: string, body = {}) => api<T>(path, { method: 'POST', body: JSON.stringify(body) });
const time = (value: string) => new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
const backendName = (id: string) => ({ bm25: 'BM25', tfidf: 'TF-IDF', vector: '向量检索', hybrid: '混合检索', dify: 'Dify' }[id] || id);
const prompts = ['你能根据这个知识库帮我做什么？', '这份资料中有哪些入门操作？'];
const defaultProfile: KnowledgeProfile = { role: '知识库使用助手', scope: '帮助用户理解当前知识库中的资料并解决相关问题。', context_fields: '相关对象、版本、操作阶段、错误信息', clarification_policy: '信息足够就直接回答；确实影响答案时，只询问一个关键条件。', response_style: '先给结论，再给简短步骤和验证方法。', aliases: '' };

function App() {
  const [page, setPage] = useState<Page>('chat');
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [current, setCurrent] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [backend, setBackend] = useState('bm25');
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<Session | null>(null);
  const [sessionDeleting, setSessionDeleting] = useState(false);
  const [sessionDeleteError, setSessionDeleteError] = useState('');
  const [diagnostic, setDiagnostic] = useState('');
  const [call, setCall] = useState<'idle' | 'connecting' | 'connected' | 'reconnecting'>('idle');
  const [muted, setMuted] = useState(false);
  const [agentState, setAgentState] = useState('listening');
  const [userState, setUserState] = useState('listening');
  const [liveText, setLiveText] = useState('');
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [selected, setSelected] = useState<Message | null>(null);
  const [sourceModal, setSourceModal] = useState<Message | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<Knowledge[]>([]);
  const [knowledgeId, setKnowledgeId] = useState('default');
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [editingKnowledge, setEditingKnowledge] = useState<Knowledge | null>(null);
  const [knowledgeToDelete, setKnowledgeToDelete] = useState<Knowledge | null>(null);
  const [knowledgeDeleting, setKnowledgeDeleting] = useState(false);
  const [knowledgeDeleteError, setKnowledgeDeleteError] = useState('');
  const knowledgeRef = useRef('default');
  const [openDoc, setOpenDoc] = useState<Doc | null>(null);
  const [editingDoc, setEditingDoc] = useState(false);
  const [docDraft, setDocDraft] = useState('');
  const [docBusy, setDocBusy] = useState(false);
  const [docNotice, setDocNotice] = useState('');
  const [docError, setDocError] = useState('');
  const [deleteDoc, setDeleteDoc] = useState<Doc | null>(null);
  const [search, setSearch] = useState('');
  const [searchResult, setSearchResult] = useState<{ chunks: Citation[]; latency_ms: number; backend: string } | null>(null);
  const [searching, setSearching] = useState(false);
  const roomRef = useRef<Room | null>(null);
  const activeId = useRef<string | null>(null);
  const attempt = useRef(0);
  const audioElements = useRef<HTMLElement[]>([]);
  const endRef = useRef(false);
  const bottom = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  function refreshStatus() { api<Status>('/status').then(setStatus).catch(() => setError('无法连接 Python 服务，请检查后端是否启动。')); }
  function refreshSessions() { api<Session[]>('/sessions').then(setSessions).catch(() => {}); }
  function refreshKnowledge() { return api<Knowledge[]>('/knowledge-bases').then(setKnowledgeBases).catch(e => setError(e.message)); }
  function refreshDocs(id = knowledgeRef.current) { api<Doc[]>('/documents?knowledge_base_id=' + encodeURIComponent(id)).then(data => { if (knowledgeRef.current === id) setDocs(data); }).catch(e => setError(e.message)); }
  useEffect(() => { knowledgeRef.current = knowledgeId; setSearchResult(null); setDocs([]); refreshDocs(knowledgeId); }, [knowledgeId]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const s = await api<Status>('/status');
        if (!mounted) return;
        setStatus(s); setBackend(s.default_backend || "bm25");
        const bases = await api<Knowledge[]>('/knowledge-bases');
        if (!mounted) return;
        setKnowledgeBases(bases);
        const list = await api<Session[]>('/sessions');
        if (!mounted) return;
        setSessions(list);
        const last = localStorage.getItem('support_session');
        const target = list.find(x => x.id === last);
        if (target) {
          const detail = await api<Session>('/sessions/' + target.id);
          if (!mounted) return;
          setCurrent(detail); setMessages(detail.messages || []); setBackend(detail.backend); activeId.current = detail.id;
          const boundId = bases.some(k => k.id === detail.knowledge_base_id) ? detail.knowledge_base_id : 'default';
          knowledgeRef.current = boundId; setKnowledgeId(boundId);
        }
        refreshDocs();
      } catch (e) { if (mounted) setError((e as Error).message); }
    })();
    const interval = window.setInterval(refreshStatus, 8000);
    const unload = () => {
      roomRef.current?.disconnect(true);
      if (activeId.current && roomRef.current) navigator.sendBeacon('/api/sessions/' + activeId.current + '/end', '{}');
    };
    window.addEventListener('beforeunload', unload);
    return () => { mounted = false; window.clearInterval(interval); window.removeEventListener('beforeunload', unload); roomRef.current?.disconnect(true); };
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, liveText]);
  useEffect(() => {
    if (call !== 'connected') return;
    const timer = window.setInterval(() => setSeconds(x => x + 1), 1000);
    return () => window.clearInterval(timer);
  }, [call]);

  async function newSession(nextBackend = backend, nextKnowledge = knowledgeId) {
    const next = await post<Session>('/sessions', { backend: nextBackend, knowledge_base_id: nextKnowledge });
    activeId.current = next.id; setCurrent(next); setMessages([]); setSelected(null); setLiveText('');
    setBackend(nextBackend); localStorage.setItem('support_session', next.id); refreshSessions();
    knowledgeRef.current = nextKnowledge; setKnowledgeId(nextKnowledge);
    return next;
  }
  async function selectSession(s: Session) {
    if (call !== 'idle' || busy) return;
    try {
      const data = await api<Session>('/sessions/' + s.id);
      activeId.current = data.id; setCurrent(data); setMessages(data.messages || []); setBackend(data.backend);
      const boundId = knowledgeBases.some(k => k.id === data.knowledge_base_id) ? data.knowledge_base_id : 'default';
      knowledgeRef.current = boundId; setKnowledgeId(boundId);
      setSelected(null); setPage('chat'); setError(''); localStorage.setItem('support_session', data.id);
    } catch (e) { setError((e as Error).message); }
  }
  async function changeBackend(value: string) {
    if (call !== 'idle' || busy) return;
    try { await newSession(value); setError(''); } catch (e) { setError((e as Error).message); }
  }
  function confirmSessionDelete(session: Session) {
    if (call !== 'idle' || busy) return;
    setHistoryOpen(false); setSessionDeleteError(''); setSessionToDelete(session);
  }
  async function removeSession() {
    if (!sessionToDelete || sessionDeleting || busy || call !== 'idle') return;
    const id = sessionToDelete.id;
    setSessionDeleting(true); setBusy(true); setSessionDeleteError('');
    try {
      // Clean up a call left active by a previous browser/tab before removing its history.
      if (sessionToDelete.status === 'active') await post('/sessions/' + id + '/end');
      await api('/sessions/' + id, { method: 'DELETE' });
      setSessions(items => items.filter(item => item.id !== id));
      if (activeId.current === id) {
        activeId.current = null; setCurrent(null); setMessages([]); setSelected(null);
        setSourceModal(null); setLiveText(''); setInput(''); setSeconds(0);
        localStorage.removeItem('support_session');
      }
      setSessionToDelete(null); refreshSessions();
    } catch (e) { setSessionDeleteError((e as Error).message); }
    finally { setSessionDeleting(false); setBusy(false); }
  }
  async function changeKnowledge(value: string) {
    if (call !== 'idle' || busy || docBusy) return;
    setBusy(true); setError('');
    try { await newSession(backend, value); setDocNotice(''); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function knowledgeSaved(value: Knowledge) {
    await refreshKnowledge();
    if (!editingKnowledge) await newSession(backend, value.id);
    setDocNotice('知识库配置已保存，新会话会使用这份配置。');
  }
  async function removeKnowledge() {
    if (!knowledgeToDelete || knowledgeDeleting || busy || docBusy || call !== 'idle') return;
    const id = knowledgeToDelete.id;
    setKnowledgeDeleting(true); setBusy(true); setKnowledgeDeleteError('');
    try {
      await api('/knowledge-bases/' + id, { method: 'DELETE' });
      setKnowledgeBases(items => items.filter(k => k.id !== id));
      if (current?.knowledge_base_id === id) setCurrent({ ...current, status: 'ended' });
      if (knowledgeRef.current === id) {
        knowledgeRef.current = 'default'; setKnowledgeId('default');
        setDocs([]); setSearchResult(null); setOpenDoc(null); setDeleteDoc(null);
      }
      setKnowledgeToDelete(null); setDocNotice('知识库及其文档、向量已删除，历史会话和引用保留供查看。');
      refreshSessions(); refreshStatus(); refreshKnowledge();
    } catch (e) { setKnowledgeDeleteError((e as Error).message); }
    finally { setKnowledgeDeleting(false); setBusy(false); }
  }
  async function sendText(text: string) {
    if (!text.trim() || busy || call !== 'idle') return;
    setError(''); setBusy(true); setInput('');
    try {
      const session = !current || current.status === 'ended' ? await newSession() : current;
      const id = session.id;
      const optimistic: Message = { id: 'pending-' + Date.now(), session_id: id, turn_id: '', role: 'user', text,
        citations: [], metrics: {}, interrupted: false, answer_mode: 'text', created_at: new Date().toISOString() };
      setMessages(m => [...m, optimistic]);
      const reply = await post<Message>('/sessions/' + id + '/chat', { text, preview: !status?.llm_configured });
      if (activeId.current === id) {
        const detail = await api<Session>('/sessions/' + id); setMessages(detail.messages || []); setCurrent(detail); setSelected(reply);
      }
      refreshSessions();
    } catch (e) {
      setError((e as Error).message);
      if (activeId.current) api<Session>('/sessions/' + activeId.current).then(d => setMessages(d.messages || [])).catch(() => {});
    } finally { setBusy(false); }
  }
  function receiveMessage(message: Message) {
    if (message.session_id !== activeId.current) return;
    setMessages(old => old.some(x => x.id === message.id) ? old.map(x => x.id === message.id ? message : x) : [...old, message]);
    setLiveText('');
    if (message.role === 'assistant') setSelected(message);
    refreshSessions();
  }
  async function cleanupRoom() {
    const room = roomRef.current; roomRef.current = null;
    if (room) {
      room.localParticipant.audioTrackPublications.forEach(p => p.track?.stop());
      await room.disconnect(true);
    }
    audioElements.current.forEach(e => e.remove()); audioElements.current = [];
    setCall('idle'); setMuted(false); setLiveText(''); setAudioBlocked(false);
  }
  async function hangup() {
    attempt.current++; endRef.current = true;
    await cleanupRoom();
    const id = activeId.current;
    if (id) {
      try { await post('/sessions/' + id + '/end');
        const detail = await api<Session>('/sessions/' + id); setCurrent(detail); setMessages(detail.messages || []);
      } catch (e) { setError((e as Error).message); }
    }
    refreshSessions(); refreshStatus();
  }
  async function startCall() {
    if (call !== 'idle' || busy) return;
    if (!status?.voice_ready) { setSettingsOpen(true); refreshStatus(); return; }
    const tokenAttempt = ++attempt.current;
    setCall('connecting'); setError(''); setSeconds(0); endRef.current = false;
    try {
      const target = !current || current.status === 'ended' || messages.length ? await newSession() : current;
      if (tokenAttempt !== attempt.current) { await post("/sessions/" + target.id + "/end"); return; }
      const token = await post<{ url: string; token: string }>('/sessions/' + target.id + '/token');
      if (tokenAttempt !== attempt.current) { await post("/sessions/" + target.id + "/end"); return; }
      const room = new Room({ singlePeerConnection: false, adaptiveStream: true, dynacast: true, audioCaptureDefaults: {
        echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      } });
      roomRef.current = room;
      room.registerTextStreamHandler('support.events', async reader => {
        const payload = JSON.parse(await reader.readAll());
        if (tokenAttempt !== attempt.current) { await post("/sessions/" + target.id + "/end"); return; }
        if (payload.type === 'message') receiveMessage(payload.message);
        if (payload.type === 'agent_state') setAgentState(payload.state);
        if (payload.type === 'user_state') setUserState(payload.state);
        if (payload.type === 'error') setError(payload.detail);
      });
      room.registerTextStreamHandler('lk.transcription', async (reader, info) => {
        let text = '';
        for await (const chunk of reader) {
          text += chunk;
          if (tokenAttempt === attempt.current) setLiveText((info.identity.startsWith('user-') ? '你：' : '知声：') + text);
        }
      });
      room.on(RoomEvent.TrackSubscribed, track => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach(); el.setAttribute('data-support-audio', 'true'); document.body.appendChild(el); audioElements.current.push(el);
        }
      });
      room.on(RoomEvent.TrackUnsubscribed, track => track.detach().forEach(el => el.remove()));
      room.on(RoomEvent.AudioPlaybackStatusChanged, () => setAudioBlocked(!room.canPlaybackAudio));
      room.on(RoomEvent.Reconnecting, () => setCall('reconnecting'));
      room.on(RoomEvent.Reconnected, () => setCall('connected'));
      room.on(RoomEvent.Disconnected, () => {
        if (!endRef.current && tokenAttempt === attempt.current) {
          setError('通话连接已断开。记录已保存，可以新建通话重试。'); void hangup();
        }
      });
      await room.connect(token.url, token.token);
      if (tokenAttempt !== attempt.current) { await room.disconnect(true); return; }
      await room.startAudio();
      await room.localParticipant.setMicrophoneEnabled(true);
      if (tokenAttempt !== attempt.current) { room.localParticipant.audioTrackPublications.forEach(p => p.track?.stop()); await room.disconnect(true); return; }
      setCall('connected'); refreshStatus();
      window.setTimeout(() => {
        if (tokenAttempt === attempt.current && room.remoteParticipants.size === 0) {
          setError('Agent 尚未加入房间，请检查 Agent 日志后重试。'); void hangup();
        }
      }, 45000);
    } catch (e) {
      const name = (e as Error).name;
      setError(name === 'NotAllowedError' ? '麦克风权限被拒绝。请在浏览器地址栏允许麦克风，然后重试。' : (e as Error).message);
      await hangup();
    }
  }
  async function toggleMute() {
    if (!roomRef.current || call !== 'connected') return;
    try { await roomRef.current.localParticipant.setMicrophoneEnabled(muted); setMuted(!muted); }
    catch (e) { setError((e as Error).message); }
  }
  async function readDoc(id: string) {
    try { setEditingDoc(false); setDocError(''); setOpenDoc(await api<Doc>('/documents/' + id)); } catch (e) { setError((e as Error).message); }
  }
  async function diagnoseConnection() {
    if (activeCall || diagnostic === '正在测试连接…') return;
    setDiagnostic('正在测试连接…');
    const room = new Room({ singlePeerConnection: false });
    let sid: string | null = null;
    try {
      const test = await post<Session>('/sessions', { backend: 'bm25', diagnostic: true, knowledge_base_id: knowledgeId });
      sid = test.id;
      const token = await post<{ url: string; token: string }>('/sessions/' + sid + '/token');
      await room.connect(token.url, token.token);
      setDiagnostic('连接成功：WebRTC 已建立，未启用麦克风或播放声音。');
    } catch (e) {
      setDiagnostic('连接测试失败：' + (e as Error).message);
    } finally {
      await room.disconnect(true);
      if (sid) await post('/sessions/' + sid + '/end').catch(() => {});
      refreshSessions();
    }
  }
  async function upload(file: File) {
    setDocBusy(true); setDocNotice(''); setError('');
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error('单份文档不能超过 10 MB。');
      const data = new FormData(); data.append('file', file); data.append('knowledge_base_id', knowledgeId);
      await api('/documents/upload', { method: 'POST', body: data });
      refreshDocs(); refreshStatus(); refreshKnowledge(); setSearchResult(null);
      setDocNotice(`“${file.name}”已完成解析和索引，可以直接提问。`);
    } catch (e) { setError((e as Error).message); }
    finally { setDocBusy(false); }
  }
  async function saveDoc() {
    if (!openDoc || docBusy) return;
    setDocBusy(true); setError(''); setDocError('');
    try {
      await api('/documents/' + openDoc.id, { method: 'PUT', body: JSON.stringify({ content: docDraft, expected_version: openDoc.version }) });
      await readDoc(openDoc.id); refreshDocs(); refreshStatus(); setSearchResult(null);
      setDocNotice('文档已保存，检索将使用新版本。');
    } catch (e) { setDocError((e as Error).message); }
    finally { setDocBusy(false); }
  }
  async function removeDoc() {
    if (!deleteDoc || docBusy) return;
    setDocBusy(true); setError(''); setDocError('');
    try {
      await api('/documents/' + deleteDoc.id, { method: 'DELETE' });
      setOpenDoc(null); setDeleteDoc(null); refreshDocs(); refreshStatus(); refreshKnowledge(); setSearchResult(null);
      setDocNotice('文档及其向量已移出知识库，后续检索不再使用。');
    } catch (e) { setDocError((e as Error).message); }
    finally { setDocBusy(false); }
  }
  async function testSearch(event: FormEvent) {
    event.preventDefault(); if (!search.trim() || searching) return;
    setSearching(true); setError('');
    try { setSearchResult(await post('/search', { query: search, backend, knowledge_base_id: knowledgeId })); }
    catch (e) { setError((e as Error).message); } finally { setSearching(false); }
  }
  const activeCall = call !== 'idle';
  const callLabel = call === 'connecting' ? '正在连接' : call === 'reconnecting' ? '正在重连' : muted ? '麦克风已静音' : userState === 'speaking' ? '正在听你说' : agentState === 'thinking' ? '正在查找资料' : agentState === 'speaking' ? '知声正在回答' : '正在聆听';
  const latestAnswer = [...messages].reverse().find(m => m.role === 'assistant' && m.answer_mode !== 'error');
  const sourceMessage = selected || latestAnswer;
  const turns = messages.filter(m => m.role === 'user').length;
  const totalLatency = latestAnswer?.metrics.total_ms || (latestAnswer?.metrics.e2e_latency ? latestAnswer.metrics.e2e_latency * 1000 : null);
  const navItems = ([{ id: 'chat', label: '语音助手', Icon: Headphones }, { id: 'knowledge', label: '知识库', Icon: BookOpen }, { id: 'metrics', label: '运行概览', Icon: Activity }] as const);
  const pageTitle = page === 'chat' ? '知识库语音助手' : page === 'knowledge' ? '知识库' : '运行概览';
  const selectedKnowledge = knowledgeBases.find(k => k.id === knowledgeId);
  const boundKnowledge = current?.knowledge_profile?.name ? current.knowledge_profile : selectedKnowledge;

  return <TooltipProvider>
    <div className="flex h-dvh overflow-hidden bg-muted/40 text-foreground">
      <aside className="hidden w-64 shrink-0 flex-col border-r bg-sidebar md:flex">
        <div className="flex h-14 shrink-0 items-center gap-2.5 border-b px-5">
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm"><AudioLines size={18} /></span>
          <span className="flex flex-col leading-none">
            <span className="text-[15px] font-semibold tracking-tight">知声</span>
            <span className="mt-1 text-[10px] font-medium tracking-[0.2em] text-muted-foreground">SUPPORT STUDIO</span>
          </span>
        </div>
        <div className="p-3">
          <Button className="w-full justify-start gap-2" disabled={activeCall || busy || !status} onClick={() => { newSession().then(() => { setPage('chat'); setError(''); }).catch(e => setError(e.message)); }}><Plus size={16} />新建会话</Button>
        </div>
        <div className="px-5 pb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">工作空间</div>
        <nav className="flex flex-col gap-1 px-3">{navItems.map(({ id, label, Icon }) => <Button key={id} variant={page === id ? 'secondary' : 'ghost'} className={cn('h-9 justify-start gap-2.5 px-3 font-medium', page === id && 'bg-primary/10 text-primary hover:bg-primary/15')} onClick={() => setPage(id)}><Icon size={17} />{label}{id === 'knowledge' && <Badge variant="secondary" className="ml-auto h-5 min-w-5 justify-center px-1.5 text-[10px] font-semibold">{status?.document_count || '—'}</Badge>}</Button>)}</nav>
        <div className="mt-5 flex min-h-0 flex-1 flex-col px-5">
          <div className="flex items-center justify-between pb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"><span>最近会话</span><Clock3 size={12} /></div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="flex flex-col gap-0.5 pr-3">
              {sessions.length ? sessions.map(s => <div key={s.id} className={cn('flex items-center rounded-lg transition-colors hover:bg-accent', current?.id === s.id ? 'bg-primary/10 text-primary' : 'text-foreground')}>
                <button disabled={activeCall || busy} className="flex min-w-0 flex-1 items-start gap-2.5 px-2.5 py-2 text-left disabled:pointer-events-none disabled:opacity-50" onClick={() => selectSession(s)}><MessageSquare size={14} className="mt-0.5 shrink-0 opacity-70" /><span className="min-w-0 flex-1 text-xs font-medium leading-snug"><span className="line-clamp-2 block">{s.title}</span><small className="mt-1 block text-[10px] font-normal text-muted-foreground">{time(s.updated_at)} · {backendName(s.backend)}</small></span></button>
                <Button variant="ghost" size="icon" className="mr-1 size-7 shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive" disabled={activeCall || busy} title="删除会话" aria-label={'删除会话：' + s.title} onClick={() => confirmSessionDelete(s)}><Trash2 size={13} /></Button>
              </div>) : <p className="px-2.5 py-8 text-center text-xs text-muted-foreground">你的咨询记录会保存在这里</p>}
            </div>
          </ScrollArea>
        </div>
        <div className="space-y-3 border-t p-3">
          <div className="flex items-center gap-2.5 rounded-xl border bg-background p-2.5 shadow-sm">
            <span className={cn('size-2 shrink-0 rounded-full', status?.livekit_ready ? 'bg-emerald-500' : 'bg-muted-foreground/30')} />
            <div className="min-w-0 flex-1 leading-tight"><span className="block text-xs font-medium">{status?.deployment === 'server' ? '服务器运行' : '本地运行'}</span><small className="block text-[10px] text-muted-foreground">LiveKit WebRTC 音频传输</small></div>
            <ShieldCheck size={15} className="shrink-0 text-emerald-600" />
          </div>
          <Button variant="ghost" className="h-8 w-full justify-between px-2.5 text-xs font-normal text-muted-foreground" onClick={() => setSettingsOpen(true)}><span className="flex items-center gap-2"><Settings2 size={15} />服务配置</span><ChevronRight size={13} /></Button>
          <div className="flex items-center gap-2.5 rounded-xl border bg-background p-2.5 shadow-sm">
            <Avatar className="size-8"><AvatarFallback className="bg-primary/10 text-xs text-primary">访</AvatarFallback></Avatar>
            <div className="min-w-0 flex-1 leading-tight"><span className="block truncate text-xs font-medium">{status?.deployment === 'server' ? '服务器演示空间' : '本地演示空间'}</span><small className="block text-[10px] text-muted-foreground">LiveKit Agents · Python</small></div>
            <span className="text-[10px] text-muted-foreground">v0.1</span>
          </div>
        </div>
      </aside>
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b bg-card/80 px-4 backdrop-blur md:px-6">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className="hidden text-xs text-muted-foreground sm:inline">工作空间</span>
            <ChevronRight size={12} className="hidden text-muted-foreground sm:inline" />
            <h1 className="truncate text-sm font-semibold tracking-tight">{pageTitle}</h1>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Select value={knowledgeId} onValueChange={changeKnowledge} disabled={activeCall || busy || docBusy}>
              <SelectTrigger className="w-52" aria-label="当前知识库"><SelectValue placeholder="选择知识库" /></SelectTrigger>
              <SelectContent>{knowledgeBases.map(k => <SelectItem key={k.id} value={k.id}>{k.name}</SelectItem>)}</SelectContent>
            </Select>
            {activeCall && <button className="flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700 transition-colors hover:bg-emerald-100" onClick={hangup} aria-label="结束当前通话"><span className="size-1.5 rounded-full bg-emerald-500" /><span className="hidden sm:inline">通话中</span><PhoneOff size={12} /></button>}
            <Badge variant="outline" className="gap-1.5 px-2 py-0.5 text-[10px] font-medium text-muted-foreground"><span className={cn('size-1.5 rounded-full', status?.livekit_ready ? 'bg-emerald-500' : 'bg-muted-foreground/30')} />LOCAL DEMO</Badge>
            <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="size-8 rounded-full text-muted-foreground" title="服务配置与帮助" aria-label="服务配置与帮助" onClick={() => setSettingsOpen(true)}><CircleHelp size={18} /></Button></TooltipTrigger><TooltipContent>服务配置与帮助</TooltipContent></Tooltip>
          </div>
        </header>
        {error && <div className="px-4 pt-3 md:px-6"><Alert variant="destructive" role="alert" className="flex items-center gap-2 border-destructive/30 bg-destructive/5 py-2.5 text-destructive [&>svg]:top-2.5"><CircleHelp size={15} className="shrink-0" /><AlertDescription className="flex min-w-0 flex-1 items-center text-xs">{error}</AlertDescription><Button variant="ghost" size="icon" className="size-6 shrink-0 text-destructive hover:bg-destructive/10 hover:text-destructive" onClick={() => { setError(''); refreshStatus(); }} aria-label="关闭错误提示"><X size={14} /></Button></Alert></div>}
        {page === 'chat' && <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 md:p-6 lg:flex-row lg:overflow-hidden">
          <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card shadow-sm">
            <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b px-4">
              <div className="flex min-w-0 items-center gap-2">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary"><Sparkles size={13} /></span>
                <strong className="text-sm font-semibold">知声</strong>
                <span className="truncate text-xs text-muted-foreground">{boundKnowledge?.name || '知识库助手'}</span>
              </div>
              <Label className="flex shrink-0 items-center gap-2 text-[11px] text-muted-foreground">
                <span className="hidden sm:inline">检索引擎</span>
                <Select value={backend} onValueChange={value => changeBackend(value)} disabled={activeCall || busy}>
                  <SelectTrigger className="h-7 w-28 gap-1 rounded-full border-border bg-background px-2.5 text-[11px] shadow-none" aria-label="检索引擎"><SelectValue placeholder={backendName(backend)} /></SelectTrigger>
                  <SelectContent>{(status?.backends || [{ id: 'bm25', name: 'BM25', ready: true }]).map(b => <SelectItem key={b.id} value={b.id} disabled={!b.ready}>{b.name}{!b.ready ? '（未配置）' : ''}</SelectItem>)}</SelectContent>
                </Select>
              </Label>
            </div>
            <ScrollArea className="min-h-0 flex-1" aria-live="polite">
              <div className="flex min-h-full flex-col justify-end p-4">
                {!messages.length && <div className="flex flex-1 flex-col items-center justify-center py-8 text-center">
                  <div className="mb-6 flex items-center gap-2 rounded-full border bg-background px-3 py-1 text-[11px] font-medium text-muted-foreground shadow-sm"><span className="size-1.5 rounded-full bg-emerald-500" />连接问题与答案</div>
                  <div className="relative mb-7 flex size-24 items-center justify-center">
                    <span className="orb-ring absolute inset-0 rounded-full border-2 border-primary/25" />
                    <span className="orb-float flex size-16 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-xl shadow-primary/25"><AudioLines size={28} strokeWidth={1.5} /></span>
                  </div>
                  <h2 className="text-2xl font-semibold leading-snug tracking-tight">说出问题，<br />答案有据可查。</h2>
                  <p className="mt-3 max-w-md text-sm leading-relaxed text-muted-foreground">{boundKnowledge?.profile?.scope || '选择知识库，上传资料后，就可以开始对话。'}</p>
                  <div className="mt-5 flex flex-wrap justify-center gap-2">
                    <Badge variant="secondary" className="gap-1 px-2.5 py-1 text-[11px] font-normal"><Mic size={11} />自然对话</Badge>
                    <Badge variant="secondary" className="gap-1 px-2.5 py-1 text-[11px] font-normal"><Zap size={11} />随时打断</Badge>
                    <Badge variant="secondary" className="gap-1 px-2.5 py-1 text-[11px] font-normal"><BookOpen size={11} />来源可查</Badge>
                  </div>
                  <div className="mt-8 w-full max-w-md space-y-2 text-left">
                    <span className="block text-center text-xs text-muted-foreground">也可以从这些问题开始</span>
                    {prompts.map((p, i) => <Button key={p} variant="outline" disabled={busy || activeCall || !status} onClick={() => sendText(p)} className="h-auto w-full justify-start gap-2.5 rounded-xl px-3 py-2.5 text-left text-[13px] font-normal shadow-sm"><span className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">{i === 0 ? <Zap size={14} /> : i === 1 ? <Settings2 size={14} /> : <Cable size={14} />}</span><span className="flex-1">{p}</span><ArrowUpRight size={14} className="shrink-0 text-muted-foreground" /></Button>)}
                  </div>
                </div>}
                <div className="flex flex-col gap-4">
                  {messages.map(m => <article key={m.id} className={cn('flex gap-2.5', m.role === 'user' && 'flex-row-reverse')}>
                    <Avatar className="mt-0.5 size-8 shrink-0"><AvatarFallback className={cn('text-[11px]', m.role === 'assistant' ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground')}>{m.role === 'assistant' ? <AudioLines size={15} /> : '你'}</AvatarFallback></Avatar>
                    <div className={cn('flex min-w-0 max-w-[86%] flex-col gap-1.5', m.role === 'user' && 'items-end')}>
                      <div className={cn('flex items-center gap-2 text-[11px] text-muted-foreground', m.role === 'user' && 'flex-row-reverse')}>
                        <strong className="text-[11px] font-semibold text-foreground">{m.role === 'assistant' ? '知声' : '你'}</strong>
                        <time>{time(m.created_at)}</time>
                        {m.interrupted && <Badge variant="outline" className="h-[18px] px-1.5 text-[10px] font-normal">已打断</Badge>}
                        {m.answer_mode === 'extractive' && <Badge variant="secondary" className="h-[18px] px-1.5 text-[10px] font-normal">本地资料摘录</Badge>}
                      </div>
                      <div className={cn('rounded-2xl px-3.5 py-2.5 text-[13px] leading-relaxed shadow-sm', m.role === 'assistant' ? 'rounded-tl-md border bg-muted/60' : 'rounded-tr-md bg-primary text-primary-foreground', m.answer_mode === 'error' && 'border-destructive/25 bg-destructive/5 text-destructive')}>{m.text}</div>
                      {m.role === 'assistant' && m.citations.length > 0 && <Button variant="outline" size="sm" className={cn('h-7 gap-1.5 rounded-full px-2.5 text-[11px] font-normal text-muted-foreground shadow-none', sourceMessage?.id === m.id && 'border-primary/40 bg-primary/5 text-primary')} onClick={() => { setSelected(m); if (window.matchMedia("(max-width: 630px)").matches) setSourceModal(m); }}><BookOpen size={12} />{m.citations.length} 条来源引用<ChevronRight size={12} />{m.metrics.retrieval_ms != null && <span className="text-[10px] opacity-70">{m.metrics.retrieval_ms.toFixed(0)} ms 检索</span>}</Button>}
                    </div>
                  </article>)}
                  {busy && <div className="flex items-center gap-2 px-1 text-[11px] text-muted-foreground"><span className="flex items-end gap-1"><span className="thinking-dot size-1.5 rounded-full bg-primary" /><span className="thinking-dot size-1.5 rounded-full bg-primary" style={{ animationDelay: '0.15s' }} /><span className="thinking-dot size-1.5 rounded-full bg-primary" style={{ animationDelay: '0.3s' }} /></span>正在查阅知识库资料</div>}
                  {activeCall && liveText && <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-xs text-emerald-800"><span className="pulse-dot size-2 shrink-0 rounded-full bg-emerald-500" />{liveText}</div>}
                  <div ref={bottom} />
                </div>
              </div>
            </ScrollArea>
            <div className="shrink-0 space-y-3 border-t bg-card p-3 md:p-4">
              {audioBlocked && <Button variant="outline" size="sm" className="w-full" onClick={() => roomRef.current?.startAudio()}>点击启用浏览器声音播放</Button>}
              {!status?.llm_configured && <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-700"><ShieldCheck size={12} className="shrink-0" />文字模式当前展示真实检索摘录，配置 LLM 后启用模型问答。</div>}
              <div className={cn('flex items-center gap-3 rounded-xl border p-3 transition-colors', activeCall ? 'border-emerald-200 bg-emerald-50/50' : 'bg-muted/50')}>
                <span className={cn('flex size-10 shrink-0 items-center justify-center rounded-full transition-colors', activeCall ? 'bg-emerald-500 text-white' : 'bg-primary/10 text-primary')}><AudioLines size={18} /></span>
                <div className="min-w-0 flex-1 leading-tight">
                  <strong className="block text-[13px] font-semibold">{activeCall ? callLabel : '准备好聊一聊了吗？'}</strong>
                  <span className="mt-0.5 block truncate text-[11px] text-muted-foreground">{activeCall ? `通话 ${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')} · 可以随时开口打断` : '打开麦克风，直接告诉我你的问题'}</span>
                </div>
                {activeCall ? <div className="flex shrink-0 items-center gap-2">
                  <div className="flex h-6 items-end gap-0.5">{Array.from({ length: 15 }, (_, i) => <i key={i} className={cn('wave-bar w-1 rounded-full bg-emerald-500/70', muted && 'paused')} style={{ animationDelay: `${i * 0.09}s`, height: `${8 + ((i * 7) % 19)}px` }} />)}</div>
                  <Button variant="outline" size="icon" className="size-9 rounded-full" disabled={call !== 'connected'} onClick={toggleMute} title={muted ? '取消静音' : '静音'} aria-label={muted ? '取消静音' : '静音'}>{muted ? <MicOff size={15} /> : <Mic size={15} />}</Button>
                  <Button variant="destructive" size="icon" className="size-9 rounded-full" onClick={hangup} title="结束通话" aria-label="结束通话"><PhoneOff size={15} /></Button>
                </div> : <Button className="shrink-0 gap-2 rounded-full" onClick={startCall}><Phone size={15} />开始通话</Button>}
              </div>
              <form className="flex items-center gap-2 rounded-xl border bg-background px-3 shadow-sm transition-shadow focus-within:ring-[3px] focus-within:ring-ring/25" onSubmit={e => { e.preventDefault(); sendText(input); }}>
                <MessageSquare size={16} className="shrink-0 text-muted-foreground" />
                <Input className="h-11 flex-1 border-0 bg-transparent shadow-none focus-visible:ring-0" aria-label="输入问题" value={input} onChange={e => setInput(e.target.value)} disabled={busy || activeCall} placeholder={activeCall ? '语音通话中，请直接说出你的问题…' : '也可以输入问题，按 Enter 发送…'} maxLength={1000} />
                <Button type="submit" size="icon" className="size-8 shrink-0 rounded-full" disabled={busy || activeCall || !input.trim()} aria-label="发送问题">{busy ? <LoaderCircle size={15} className="animate-spin" /> : <ArrowUp size={15} />}</Button>
              </form>
              <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                <span>回答基于当前知识库资料 · 重要操作请核对来源</span>
                <span className="flex items-center gap-1"><ShieldCheck size={10} />会话保存在本机</span>
              </div>
            </div>
          </section>
          <aside className="flex w-full shrink-0 flex-col overflow-hidden rounded-2xl border bg-card shadow-sm lg:h-auto lg:max-h-full lg:w-80">
            <div className="shrink-0 border-b p-4">
              <div className="flex items-center gap-2">
                <BookOpen size={15} className="text-primary" />
                <strong className="text-sm font-semibold">知识来源</strong>
                <Badge variant="secondary" className="ml-auto h-5 min-w-5 justify-center px-1.5 text-[11px] font-semibold">{sourceMessage?.citations.length || 0}</Badge>
              </div>
              <p className="mt-1 text-[11px] text-muted-foreground">每一个答案，都有迹可循。</p>
            </div>
            <ScrollArea className="min-h-0 flex-1">
              <div className="space-y-3 p-4">
                {sourceMessage?.citations.length ? <>
                  <div className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-600"><span className="size-1.5 rounded-full bg-emerald-500" />当前回答的引用资料</div>
                  {sourceMessage.citations.map((c, i) => <div className="rounded-xl border bg-background p-3 shadow-sm" key={c.chunk_id}>
                    <div className="flex items-start gap-2.5">
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><FileText size={15} /></span>
                      <div className="min-w-0 flex-1 leading-tight"><strong className="block truncate text-[13px]">{c.document_name}</strong><small className="mt-0.5 block truncate text-[11px] text-muted-foreground">{c.heading || '检索片段'}</small></div>
                      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-muted-foreground">{i + 1}</span>
                    </div>
                    <p className="mt-2.5 text-[11px] leading-relaxed text-muted-foreground">{c.text}</p>
                    <div className="mt-2.5 flex items-center justify-between border-t pt-2">
                      <span className="font-mono text-[10px] text-muted-foreground">片段 {c.chunk_id.slice(0, 6)}{c.version && ' · ' + c.version.slice(0, 6)}</span>
                      <Button variant="ghost" size="icon" className="size-6 text-muted-foreground" onClick={() => readDoc(c.document_id)} aria-label={'查看 ' + c.document_name}><ArrowUpRight size={13} /></Button>
                    </div>
                  </div>)}
                </> : <div className="flex flex-col items-center justify-center rounded-xl border border-dashed py-10 text-center">
                  <div className="mb-3 flex size-14 items-center justify-center rounded-2xl bg-muted text-muted-foreground"><FileText size={26} strokeWidth={1.2} /></div>
                  <strong className="text-[13px]">{sourceMessage ? '本条回答没有资料引用' : '来源会出现在这里'}</strong>
                  <p className="mt-1.5 max-w-[220px] text-[11px] leading-relaxed text-muted-foreground">{sourceMessage ? '澄清、资料不足或服务失败时，助手会明确说明。' : '提问后，你可以查看回答参考的文档与原文片段。'}</p>
                </div>}
                <div className="rounded-xl border bg-gradient-to-b from-primary/5 to-transparent p-3.5">
                  <div className="flex items-center gap-2"><span className="flex size-7 items-center justify-center rounded-lg bg-primary/10 text-primary"><BookOpen size={14} /></span><strong className="text-[13px]">我的知识库</strong></div>
                  <p className="mt-2 text-[11px] text-muted-foreground">{docs.length} 份文档<span className="mx-1">·</span>{docs.reduce((sum, d) => sum + d.chunks, 0)} 个知识片段</p>
                  <Progress className="mt-2.5 h-1.5" value={docs.length ? 100 : 0} />
                  <Button variant="outline" size="sm" className="mt-3 w-full justify-center gap-1 text-xs" onClick={() => setPage('knowledge')}>浏览知识库<ChevronRight size={13} /></Button>
                </div>
                <div className="flex items-start gap-2.5 rounded-xl bg-muted/60 p-3.5">
                  <ShieldCheck size={16} className="mt-0.5 shrink-0 text-emerald-600" />
                  <p className="text-[11px] leading-relaxed text-muted-foreground">资料不足时，知声会主动告知，帮助你确认下一步。</p>
                </div>
              </div>
            </ScrollArea>
          </aside>
        </div>}
        {page === 'knowledge' && <ScrollArea className="min-h-0 flex-1"><section className="mx-auto max-w-5xl space-y-5 p-4 md:p-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-primary">KNOWLEDGE BASE</span>
              <h2 className="mt-1.5 text-xl font-semibold tracking-tight">让回答建立在资料之上</h2>
              <p className="mt-1.5 text-sm text-muted-foreground">当前：{selectedKnowledge?.name}。切换知识库会新建会话，文档只在所属知识库中检索。</p>
            </div>
            <Button className="shrink-0 gap-2" disabled={docBusy} onClick={() => fileInput.current?.click()}>{docBusy ? <LoaderCircle size={15} className="animate-spin" /> : <Upload size={15} />}{docBusy ? '正在解析与索引…' : '上传文档'}</Button>
            <input ref={fileInput} type="file" accept=".md,.txt,.pdf,.docx" hidden onChange={e => { const file = e.target.files?.[0]; if (file) upload(file); e.target.value = ''; }} />
          </div>
          <Card className="gap-3 p-4">
            <div className="flex items-center gap-2"><Sparkles size={16} /><strong className="flex-1 text-sm">{selectedKnowledge?.profile.role || '助手业务配置'}</strong>
              <Button variant="outline" size="sm" disabled={activeCall || busy || docBusy} onClick={() => { setEditingKnowledge(null); setKnowledgeOpen(true); }}><Plus size={14} />新建知识库</Button>
              <Button size="sm" disabled={!selectedKnowledge || activeCall || busy || docBusy} onClick={() => { setEditingKnowledge(selectedKnowledge || null); setKnowledgeOpen(true); }}><Pencil size={14} />编辑助手说明</Button>
              <Button variant="destructive" size="sm" disabled={!selectedKnowledge || knowledgeId === 'default' || activeCall || busy || docBusy} title={knowledgeId === 'default' ? '默认知识库保留作为入口，可以删除其中的文档' : '删除当前知识库'} onClick={() => { setKnowledgeDeleteError(''); setKnowledgeToDelete(selectedKnowledge || null); }}><Trash2 size={14} />删除知识库</Button>
            </div>
            <p className="text-sm text-muted-foreground">{selectedKnowledge?.profile.scope}</p>
            <p className="text-xs text-muted-foreground">业务配置修改后对新会话生效；已创建的会话保留原配置，文档内容使用最新版本。</p>
            {knowledgeId === 'default' && <p className="text-xs text-muted-foreground">默认知识库保留作为入口，可清空其中的文档；自建知识库可整体删除。</p>}
          </Card>
          <Card className="gap-2 p-4">
            <div className="flex items-center gap-2"><Database size={16} /><strong className="text-sm">Qdrant 向量数据库</strong><Badge variant="outline">{status?.vector_ready ? '已连接' : '未就绪'}</Badge>{status?.vector_dashboard_url && <a className="ml-auto inline-flex items-center gap-1 text-xs text-primary" href={status.vector_dashboard_url} target="_blank" rel="noreferrer">打开数据库<ExternalLink size={12} /></a>}</div>
            <p className="text-xs text-muted-foreground">{status?.embedding_model || 'BGE 中文 embedding'} · {status?.embedding_dimension || 512} 维 · {status?.embedding_provider === 'dashscope' ? '阿里云向量化' : '本机 CPU 编码'}</p>
            {status?.knowledge_error && <p className="text-xs text-destructive">{status.knowledge_error}</p>}
          </Card>
          {docNotice && <p className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800" role="status">{docNotice}</p>}
          <div className="grid gap-3 sm:grid-cols-3">
            <Card className="flex-row items-center gap-3 p-4"><span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary"><BookOpen size={17} /></span><div><strong className="block text-xl font-semibold tracking-tight">{docs.length}</strong><span className="text-[11px] text-muted-foreground">知识库文档</span></div></Card>
            <Card className="flex-row items-center gap-3 p-4"><span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary"><FileText size={17} /></span><div><strong className="block text-xl font-semibold tracking-tight">{docs.reduce((sum, d) => sum + d.chunks, 0)}</strong><span className="text-[11px] text-muted-foreground">知识片段</span></div></Card>
            <Card className="flex-row items-center gap-3 p-4"><span className="flex size-10 items-center justify-center rounded-xl bg-emerald-100 text-emerald-600"><ShieldCheck size={17} /></span><div><strong className="block text-xl font-semibold tracking-tight">{status?.deployment === 'server' ? '服务器' : '本地'}</strong><span className="text-[11px] text-muted-foreground">文档存储位置</span></div></Card>
          </div>
          <Card className="gap-0 overflow-hidden p-0">
            <div className="flex items-center justify-between border-b px-4 py-3">
              <h3 className="text-sm font-semibold">全部文档</h3>
              <span className="text-[11px] text-muted-foreground">TXT / MD / PDF / DOCX · 最大 10 MB</span>
            </div>
            <div className="grid grid-cols-[1fr_72px_110px_88px] gap-2 border-b bg-muted/40 px-4 py-2 text-[11px] font-medium text-muted-foreground"><span>文档名称</span><span>知识片段</span><span className="hidden sm:inline">内容版本</span><span>状态</span></div>
            <div className="divide-y">
              {docs.length === 0 && <p className="p-8 text-center text-sm text-muted-foreground">知识库还没有文档，上传资料后即可检索和提问。</p>}
              {docs.map(d => <button className="grid w-full grid-cols-[1fr_72px_110px_88px] items-center gap-2 px-4 py-3 text-left transition-colors hover:bg-accent/50" key={d.id} onClick={() => readDoc(d.id)}>
                <span className="flex min-w-0 items-center gap-2.5"><span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><FileText size={16} /></span><span className="min-w-0"><strong className="block truncate text-[13px]">{d.title}</strong><small className="block truncate text-[10px] text-muted-foreground">{d.filename}</small></span></span>
                <span className="text-[11px] text-muted-foreground">{d.chunks} 个</span>
                <span className="hidden font-mono text-[10px] text-muted-foreground sm:inline">{d.version.slice(0, 8)}</span>
                <span><Badge variant="outline" className="gap-1 px-1.5 py-0.5 text-[10px] font-normal"><CheckCheck size={11} />{d.index_status === 'indexed' ? '已向量化' : d.index_status === 'lexical' ? '词项索引' : '待索引'}</Badge></span>
              </button>)}
            </div>
          </Card>
          <Card className="gap-0 p-4 md:p-5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary"><Search size={15} /></span>
              <h3 className="text-sm font-semibold">检索实验台</h3>
              <Badge variant="secondary" className="text-[10px] font-normal">当前引擎：{backendName(backend)}</Badge>
              <Select value={backend} onValueChange={changeBackend} disabled={activeCall || busy}><SelectTrigger className="ml-auto w-60" aria-label="检索引擎"><SelectValue /></SelectTrigger><SelectContent>{status?.backends.map(b => <SelectItem key={b.id} value={b.id} disabled={!b.ready}>{b.name}</SelectItem>)}</SelectContent></Select>
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground">用自己的话提问，查看召回原文；可比较关键词、语义向量和混合检索。</p>
            <form className="mt-3 flex flex-col gap-2 sm:flex-row" onSubmit={testSearch}>
              <Input className="sm:flex-1" aria-label="检索测试问题" value={search} onChange={e => setSearch(e.target.value)} placeholder="输入与当前知识库有关的问题" />
              <Button type="submit" className="shrink-0 gap-1.5" disabled={searching || !search.trim()}>{searching ? <LoaderCircle size={14} className="animate-spin" /> : <Search size={14} />}检索</Button>
            </form>
            {searchResult && <div className="mt-4 space-y-2">
              <small className="block text-[11px] text-muted-foreground">召回 {searchResult.chunks.length} 个片段 · {searchResult.latency_ms.toFixed(1)} ms · {backendName(searchResult.backend)}</small>
              {searchResult.chunks.length === 0 && <p className="text-[11px] text-muted-foreground">没有找到匹配片段。请补充更具体的描述，或检查当前知识库是否已有相关资料。</p>}
              {searchResult.chunks.map(c => <article className="rounded-xl border bg-background p-3" key={c.chunk_id}>
                <div className="flex items-center justify-between gap-2"><strong className="min-w-0 truncate text-[13px]">{c.document_name} / {c.heading}</strong><Badge variant="outline" className="shrink-0 font-mono text-[10px] font-normal">score {c.score.toFixed(3)}</Badge></div>
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">{c.text}</p>
              </article>)}
            </div>}
          </Card>
        </section></ScrollArea>}
        {page === 'metrics' && <ScrollArea className="min-h-0 flex-1"><section className="mx-auto max-w-5xl space-y-5 p-4 md:p-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-primary">OBSERVABILITY</span>
              <h2 className="mt-1.5 text-xl font-semibold tracking-tight">看清每一轮对话</h2>
              <p className="mt-1.5 text-sm text-muted-foreground">以下指标来自当前选中会话，耗时随实际调用更新。</p>
            </div>
            <Badge variant="outline" className="w-fit shrink-0 gap-1.5 border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-normal text-emerald-700"><span className="size-1.5 rounded-full bg-emerald-500" />本地记录</Badge>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric title="当前会话轮数" value={String(turns)} unit="轮" icon={<MessageSquare />} />
            <Metric title="最近回答延迟" value={totalLatency != null ? (totalLatency / 1000).toFixed(2) : '—'} unit="s" icon={<Zap />} />
            <Metric title="最近检索耗时" value={latestAnswer?.metrics.retrieval_ms?.toFixed(1) ?? '—'} unit="ms" icon={<Search />} />
            <Metric title="当前检索引擎" value={backendName(backend)} unit="" icon={<Cpu />} />
          </div>
          <Card className="gap-0 p-4 md:p-5">
            <h3 className="text-sm font-semibold">服务状态</h3>
            <div className="mt-3 space-y-2">
              {[{ title: 'LiveKit Server', desc: 'WebRTC 音频传输', ready: status?.livekit_ready }, { title: 'Python Agent', desc: 'LiveKit Agents SDK', ready: status?.agent_ready }, { title: 'LLM', desc: status?.llm_model || 'OpenAI 兼容接口', ready: status?.llm_configured }, { title: '知识库', desc: `${status?.document_count ?? 0} 份资料 · ${status?.chunk_count ?? 0} 个片段`, ready: !!status?.document_count }].map(s => <div className="flex items-center gap-3 rounded-xl border bg-background p-3" key={s.title}>
                <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><Radio size={16} /></span>
                <div className="min-w-0 flex-1 leading-tight"><strong className="block text-[13px]">{s.title}</strong><small className="mt-0.5 block truncate text-[11px] text-muted-foreground">{s.desc}</small></div>
                <Badge variant="outline" className={cn('shrink-0 gap-1.5 px-2 py-0.5 text-[10px] font-normal', s.ready ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'text-muted-foreground')}><span className={cn('size-1.5 rounded-full', s.ready ? 'bg-emerald-500' : 'bg-muted-foreground/30')} />{s.ready ? '已就绪' : '待配置 / 启动'}</Badge>
              </div>)}
            </div>
          </Card>
          <Card className="gap-0 p-4 md:p-5">
            <h3 className="text-sm font-semibold">本会话回答记录</h3>
            <div className="mt-3 space-y-2">
              {messages.filter(m => m.role === 'assistant').length ? messages.filter(m => m.role === 'assistant').map(m => <div className="flex items-center gap-3 rounded-xl border bg-background p-3" key={m.id}>
                <span className="shrink-0 font-mono text-[10px] text-muted-foreground">{time(m.created_at)}</span>
                <p className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">{m.text.slice(0, 85)}{m.text.length > 85 ? '…' : ''}</p>
                <Badge variant="secondary" className="shrink-0 text-[10px] font-normal">{m.citations.length} 条引用</Badge>
                <code className="shrink-0 font-mono text-[10px] text-muted-foreground">{m.metrics.total_ms ? m.metrics.total_ms.toFixed(0) + ' ms' : m.interrupted ? '已打断' : m.answer_mode}</code>
              </div>) : <p className="py-8 text-center text-[11px] text-muted-foreground">开始一次咨询后，这里会显示真实调用记录。</p>}
            </div>
          </Card>
        </section></ScrollArea>}
      </main>
      <KnowledgeEditor open={knowledgeOpen} knowledge={editingKnowledge} onOpenChange={setKnowledgeOpen} onSaved={knowledgeSaved} />
      <Dialog open={!!knowledgeToDelete} onOpenChange={open => { if (!open && !knowledgeDeleting) setKnowledgeToDelete(null); }}>
        <DialogContent><DialogHeader><DialogTitle>删除这个知识库？</DialogTitle></DialogHeader>
          <p className="text-sm text-muted-foreground">“{knowledgeToDelete?.name}”及其所有文档、历史版本和向量将永久删除。历史会话与引用保留供查看，后续不能继续使用该知识库对话。</p>
          <p className="text-xs text-muted-foreground">正在通话或生成回答时无法删除。删除后会切回默认知识库。</p>
          {knowledgeDeleteError && <p role="alert" className="text-sm text-destructive">{knowledgeDeleteError}</p>}
          <div className="flex justify-end gap-2"><Button variant="outline" disabled={knowledgeDeleting} onClick={() => setKnowledgeToDelete(null)}>取消</Button><Button variant="destructive" disabled={knowledgeDeleting} onClick={removeKnowledge}>{knowledgeDeleting ? '正在删除…' : '确认删除'}</Button></div>
        </DialogContent>
      </Dialog>
      <div className="fixed bottom-5 left-4 z-40 flex flex-col gap-2 md:hidden">
        <Button size="icon" className="size-11 rounded-full shadow-lg" disabled={activeCall || busy || !status} onClick={() => newSession().then(() => setPage("chat")).catch(e => setError(e.message))} aria-label="新建会话"><Plus size={19} /></Button>
        <Button size="icon" variant="outline" className="size-11 rounded-full bg-card shadow-lg" onClick={() => setHistoryOpen(true)} aria-label="查看历史会话"><Clock3 size={18} /></Button>
      </div>
      <Dialog open={settingsOpen} onOpenChange={open => { if (!open) setSettingsOpen(false); }}>
        <DialogContent className="max-h-[88dvh] max-w-lg overflow-y-auto" showCloseButton={false}>
          <DialogHeader>
            <div className="flex items-start gap-3">
              <span className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary"><Settings2 size={20} /></span>
              <div className="min-w-0 flex-1">
                <DialogTitle className="text-base">服务连接</DialogTitle>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">WebRTC 音频房间由 LiveKit 提供，语音与模型按服务端配置运行。<br />模型密钥由服务端的 <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">.env</code> 管理。</p>
              </div>
              <Button variant="ghost" size="icon" className="size-8 shrink-0 text-muted-foreground" aria-label="关闭服务配置" onClick={() => setSettingsOpen(false)}><X size={17} /></Button>
            </div>
          </DialogHeader>
          <div className="mt-4 space-y-2">
            <ConfigRow ok={!!status?.livekit_ready} title="LiveKit Server" detail={status?.livekit_public_url || "正在读取配置"} />
            <ConfigRow ok={!!status?.agent_ready} title="Python Agent" detail="启动脚本会同时启动 Agent 和 Web 服务" />
            <ConfigRow ok={!!status?.llm_configured} title="LLM 模型接口" detail="LLM_API_KEY / LLM_BASE_URL / LLM_MODEL" />
            <ConfigRow ok={!!status && !status.missing_resources.some(r => /SenseVoice|Kokoro|VITS|STT_|TTS_|DASHSCOPE_/.test(r))} title="中文语音服务" detail={status ? `${status.stt_model} + ${status.tts_model}` : '正在读取配置'} />
          </div>
          {!!status?.missing_resources.length && <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-[11px] leading-relaxed text-amber-700">{status.missing_resources.map(r => <p key={r}>· {r}</p>)}</div>}
          <Button variant="outline" className="mt-4 w-full gap-2 text-xs" disabled={activeCall || diagnostic === "正在测试连接…" || !status?.voice_ready} onClick={diagnoseConnection}><Cable size={14} />测试连接（不启用麦克风）</Button>
          {diagnostic && <p className="mt-2 rounded-lg bg-emerald-50 p-3 text-[11px] leading-relaxed text-emerald-700" role="status">{diagnostic}</p>}
          {status?.deployment !== 'server' && <div className="mt-4 rounded-xl border bg-muted/50 p-3">
            <span className="text-[10px] text-muted-foreground">在项目目录运行</span>
            <code className="mt-1 block rounded-md bg-background px-2.5 py-1.5 font-mono text-[11px]">.\scripts\start.ps1</code>
          </div>}
          {status?.deployment === 'server' && <form method="post" action="/logout"><Button variant="outline" className="mt-4 w-full" type="submit" disabled={activeCall}>退出演示</Button></form>}
          <div className="mt-4 flex items-center justify-between gap-3 border-t pt-4">
            <a className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground" href="https://docs.livekit.io/transport/self-hosting/local/" target="_blank" rel="noreferrer">LiveKit 官方文档<ExternalLink size={11} /></a>
            <Button className="gap-1.5" onClick={refreshStatus}>重新检查<ChevronRight size={14} /></Button>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={historyOpen} onOpenChange={open => { if (!open) setHistoryOpen(false); }}>
        <DialogContent className="max-h-[88dvh] max-w-md overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-base">历史会话</DialogTitle>
          </DialogHeader>
          <div className="mt-2 space-y-1">
            {sessions.length ? sessions.map(s => <div className="flex items-center rounded-xl border bg-background transition-colors hover:bg-accent/50" key={s.id}>
              <button className="flex min-w-0 flex-1 items-center gap-3 p-3 text-left disabled:pointer-events-none disabled:opacity-50" disabled={activeCall || busy} onClick={() => { selectSession(s); setHistoryOpen(false); }}><span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><MessageSquare size={15} /></span><span className="min-w-0 flex-1 leading-tight"><strong className="block truncate text-[13px]">{s.title}</strong><small className="mt-0.5 block text-[10px] text-muted-foreground">{time(s.updated_at)} · {backendName(s.backend)}</small></span><ChevronRight size={15} className="shrink-0 text-muted-foreground" /></button>
              <Button variant="ghost" size="icon" className="mr-2 size-8 shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive" disabled={activeCall || busy} title="删除会话" aria-label={'删除会话：' + s.title} onClick={() => confirmSessionDelete(s)}><Trash2 size={15} /></Button>
            </div>) : <p className="py-10 text-center text-[11px] text-muted-foreground">暂无历史会话</p>}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={!!sessionToDelete} onOpenChange={open => { if (!open && !sessionDeleting) setSessionToDelete(null); }}>
        <DialogContent><DialogHeader><DialogTitle>删除这条会话？</DialogTitle></DialogHeader>
          <p className="text-sm text-muted-foreground">“{sessionToDelete?.title}”及其聊天记录、历史引用将永久删除。知识库和文档会保留。</p>
          {sessionToDelete?.status === 'active' && <p className="text-sm text-muted-foreground">这条会话尚未结束，将先结束会话，再删除记录。</p>}
          {sessionDeleteError && <p role="alert" className="text-sm text-destructive">{sessionDeleteError}</p>}
          <div className="flex justify-end gap-2"><Button variant="outline" disabled={sessionDeleting} onClick={() => setSessionToDelete(null)}>取消</Button><Button variant="destructive" disabled={sessionDeleting} onClick={removeSession}>{sessionDeleting ? '正在删除…' : sessionToDelete?.status === 'active' ? '结束并删除' : '确认删除'}</Button></div>
        </DialogContent>
      </Dialog>
      <Dialog open={!!sourceModal} onOpenChange={open => { if (!open) setSourceModal(null); }}>
        <DialogContent className="max-h-[88dvh] max-w-lg overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center gap-2.5">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><BookOpen size={17} /></span>
              <DialogTitle className="text-base">本条回答的来源</DialogTitle>
            </div>
          </DialogHeader>
          <div className="mt-2 space-y-3">
            {sourceModal?.citations.map(c => <div className="rounded-xl border bg-background p-3.5" key={c.chunk_id}>
              <div className="flex items-center gap-2.5"><span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><FileText size={15} /></span><strong className="min-w-0 flex-1 truncate text-[13px]">{c.document_name} · {c.heading}</strong></div>
              <p className="mt-2.5 text-[11px] leading-relaxed text-muted-foreground">{c.text}</p>
              <div className="mt-2.5 flex items-center justify-between border-t pt-2">
                <span className="font-mono text-[10px] text-muted-foreground">片段 {c.chunk_id.slice(0, 6)}</span>
                <Button variant="ghost" size="sm" className="h-7 gap-1 text-[11px] text-muted-foreground" onClick={() => { setSourceModal(null); readDoc(c.document_id); }}>查看原文<ArrowUpRight size={13} /></Button>
              </div>
            </div>)}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={!!openDoc} onOpenChange={open => { if (!open) setOpenDoc(null); }}>
        <DialogContent className="max-h-[88dvh] max-w-2xl overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center gap-2.5">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary"><FileText size={17} /></span>
              <DialogTitle className="min-w-0 flex-1 truncate text-base">{openDoc?.title}</DialogTitle>
            </div>
          </DialogHeader>
          <p className="mt-1 text-[11px] text-muted-foreground">{openDoc?.filename} · 版本 {openDoc?.version} · {openDoc?.chunks} 个片段</p>
          {docError && <p className="text-sm text-destructive" role="alert">{docError}</p>}
          {editingDoc ? <><p className="mt-3 text-xs text-muted-foreground">编辑知识库使用的文字内容；PDF / DOCX 编辑的是提取文本。</p><textarea aria-label="文档内容" className="mt-2 h-80 w-full rounded-xl border bg-background p-4 font-mono text-sm leading-relaxed" value={docDraft} onChange={e => setDocDraft(e.target.value)} disabled={docBusy} /></> : <pre className="mt-3 max-h-[55dvh] overflow-auto rounded-xl border bg-muted/40 p-4 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{openDoc?.content}</pre>}
          <div className="mt-4 flex items-center gap-2 border-t pt-4">
            <Button variant="destructive" size="sm" className="gap-1.5" disabled={docBusy} onClick={() => setDeleteDoc(openDoc)}><Trash2 size={14} />删除文档</Button>
            <span className="flex-1" />
            {editingDoc ? <><Button variant="outline" disabled={docBusy} onClick={() => setEditingDoc(false)}>取消编辑</Button><Button className="gap-1.5" disabled={docBusy || docDraft.trim().length < 10} onClick={saveDoc}>{docBusy ? <LoaderCircle size={14} className="animate-spin" /> : <Save size={14} />}{docBusy ? '保存与索引中…' : '保存并更新索引'}</Button></> : <Button className="gap-1.5" disabled={docBusy} onClick={() => { setDocDraft(openDoc?.content || ''); setEditingDoc(true); }}><Pencil size={14} />编辑文档</Button>}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={!!deleteDoc} onOpenChange={open => { if (!open && !docBusy) setDeleteDoc(null); }}><DialogContent><DialogHeader><DialogTitle>删除这份文档？</DialogTitle></DialogHeader><p className="text-sm text-muted-foreground">“{deleteDoc?.filename}”及其向量会移出知识库，后续回答不会再检索它。已有对话中的历史引用仍会保留。</p><div className="flex justify-end gap-2"><Button variant="outline" disabled={docBusy} onClick={() => setDeleteDoc(null)}>取消</Button><Button variant="destructive" disabled={docBusy} onClick={removeDoc}>{docBusy ? '正在删除…' : '确认删除'}</Button></div></DialogContent></Dialog>
    </div>
  </TooltipProvider>;
}
function ConfigRow({ ok, title, detail }: { ok: boolean; title: string; detail: string }) {
  return <div className="flex items-center gap-3 rounded-xl border bg-background p-3">
    <span className={cn('flex size-7 shrink-0 items-center justify-center rounded-full', ok ? 'bg-emerald-100 text-emerald-600' : 'bg-muted text-muted-foreground')}>{ok ? <Check size={13} /> : <Clock3 size={13} />}</span>
    <div className="min-w-0 flex-1 leading-tight"><strong className="block text-[13px]">{title}</strong><small className="mt-0.5 block text-[11px] text-muted-foreground">{detail}</small></div>
    <span className={cn('shrink-0 text-[11px] font-medium', ok ? 'text-emerald-600' : 'text-muted-foreground')}>{ok ? '就绪' : '待就绪'}</span>
  </div>;
}

function KnowledgeEditor({ open, knowledge, onOpenChange, onSaved }: {
  open: boolean; knowledge: Knowledge | null; onOpenChange: (open: boolean) => void;
  onSaved: (knowledge: Knowledge) => Promise<void>;
}) {
  const [name, setName] = useState('');
  const [profile, setProfile] = useState<KnowledgeProfile>(defaultProfile);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  useEffect(() => {
    if (open) { setName(knowledge?.name || ''); setProfile(knowledge?.profile || { ...defaultProfile }); setError(''); setNotice(''); }
  }, [open, knowledge]);
  async function save(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const saved = await api<Knowledge>('/knowledge-bases' + (knowledge ? '/' + knowledge.id : ''), {
        method: knowledge ? 'PUT' : 'POST', body: JSON.stringify({ name: name.trim(), profile }),
      });
      await onSaved(saved); onOpenChange(false);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function draft() {
    if (!knowledge) return;
    setBusy(true); setError(''); setNotice('正在根据当前知识库资料生成草稿…');
    try {
      const result = await post<{ profile: KnowledgeProfile }>('/knowledge-bases/' + knowledge.id + '/profile-draft');
      setProfile(result.profile); setNotice('草稿已填入，检查或修改后点击保存。');
    } catch (e) { setError((e as Error).message); setNotice(''); }
    finally { setBusy(false); }
  }
  const fields: { key: keyof KnowledgeProfile; label: string; limit: number; hint?: string }[] = [
    { key: 'role', label: '助手定位', limit: 120, hint: '例如：IoT 平台使用助手、人事制度助手' },
    { key: 'scope', label: '服务范围', limit: 1200, hint: '描述希望助手帮助用户完成的任务' },
    { key: 'context_fields', label: '关键上下文', limit: 500, hint: '例如：平台版本、接入协议、操作阶段、错误信息' },
    { key: 'clarification_policy', label: '澄清策略', limit: 1200, hint: '说明什么时候需要追问，避免每轮机械询问' },
    { key: 'response_style', label: '回答方式', limit: 1200, hint: '例如：先给操作入口，再给步骤和验证方法' },
    { key: 'aliases', label: '术语说明（可选）', limit: 2000, hint: '提供本知识库的术语、缩写或同义名称，作为模型理解上下文的参考，不会自动替换用户原文' },
  ];
  return <Dialog open={open} onOpenChange={value => { if (!busy) onOpenChange(value); }}>
    <DialogContent className="max-h-[88dvh] max-w-2xl overflow-y-auto">
      <DialogHeader><DialogTitle>{knowledge ? '编辑助手说明' : '新建知识库'}</DialogTitle></DialogHeader>
      <p className="text-xs text-muted-foreground">业务配置引导助手理解和追问；事实、参数和操作步骤仍以文档证据为准。保存后对新会话生效。</p>
      {knowledge && <Button type="button" variant="outline" disabled={busy || !knowledge.document_count} onClick={draft}><Sparkles size={14} />根据资料生成草稿</Button>}
      {!knowledge && <p className="text-xs text-muted-foreground">先创建并上传资料，之后可自动生成业务配置草稿。</p>}
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {notice && <p role="status" className="text-sm text-primary">{notice}</p>}
      <form className="space-y-4" onSubmit={save}>
        <div className="space-y-1.5"><Label htmlFor="kb-name">知识库名称</Label><Input id="kb-name" value={name} onChange={e => setName(e.target.value)} maxLength={80} required disabled={busy} placeholder="例如：IoT 平台使用手册" /></div>
        {fields.map(field => <div className="space-y-1.5" key={field.key}>
          <Label htmlFor={'kb-' + field.key}>{field.label}</Label>
          <textarea id={'kb-' + field.key} className="min-h-20 w-full rounded-md border bg-background p-2.5 text-sm" value={profile[field.key]} maxLength={field.limit} disabled={busy} placeholder={field.hint} onChange={e => setProfile(p => ({ ...p, [field.key]: e.target.value }))} />
          {field.hint && <p className="text-[11px] text-muted-foreground">{field.hint}</p>}
        </div>)}
        <div className="flex justify-end gap-2 border-t pt-4"><Button type="button" variant="outline" disabled={busy} onClick={() => onOpenChange(false)}>取消</Button><Button type="submit" disabled={busy || !name.trim()}>{busy ? '处理中…' : knowledge ? '保存配置' : '创建知识库'}</Button></div>
      </form>
    </DialogContent>
  </Dialog>;
}
function Metric({ title, value, unit, icon }: { title: string; value: string; unit: string; icon: React.ReactNode }) {
  return <Card className="gap-0 p-4">
    <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary [&_svg]:size-4">{icon}</span>
    <p className="mt-3 text-[11px] text-muted-foreground">{title}</p>
    <strong className="mt-0.5 block text-2xl font-semibold tracking-tight">{value}<small className="ml-1 text-[11px] font-normal text-muted-foreground">{unit}</small></strong>
  </Card>;
}

createRoot(document.getElementById('root')!).render(<App />);
