"""Agent服务"""
import logging
from pathlib import Path
from typing import Dict, Optional

import json
from agno.agent import Agent
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.models.openai import OpenAIChat
from agno.embedder.jina import JinaEmbedder

logger = logging.getLogger(__name__)

class VideoNotesAgentService:
    """视频笔记Agent服务"""

    embedder = JinaEmbedder()

    def __init__(self):
        self.agent_sessions: Dict[str, Agent] = {}
        self.knowledge_bases: Dict[str, TextKnowledgeBase] = {}

    def _get_vector_db(self, task_id: str) -> LanceDb:
        """为每个task_id创建独立的向量数据库"""
        return LanceDb(
            table_name=f"notes_{task_id}",
            uri=f"/tmp/lancedb/{task_id}",
            search_type=SearchType.vector,
            embedder=JinaEmbedder(),
        )

    def _convert_json_to_text(self, json_data: dict) -> str:
        """将JSON笔记数据转换为可搜索的文本格式"""
        text_parts = []

        # 添加视频信息
        video_info = json_data.get("video_info", {})
        text_parts.append(f"视频文件: {video_info.get('source_video', 'unknown')}")
        text_parts.append(f"总时间段数: {video_info.get('total_segments', 0)}")
        text_parts.append(f"生成时间: {video_info.get('generated_at', 'unknown')}")
        text_parts.append("")

        # 添加每个时间段的内容
        segments = json_data.get("segments", [])
        for segment in segments:
            segment_id = segment.get("segment_id", "unknown")
            start_time = segment.get("start_time", "00:00:00")
            end_time = segment.get("end_time", "00:00:00")
            duration = segment.get("duration_seconds", 0)
            summary = segment.get("summary", "")
            frame_count = segment.get("frame_count", 0)

            text_parts.append(f"=== 时间段 {segment_id} ===")
            text_parts.append(f"时间范围: {start_time} - {end_time}")
            text_parts.append(f"持续时长: {duration:.1f}秒")
            text_parts.append(f"关键帧数量: {frame_count}")
            text_parts.append("")
            text_parts.append("内容摘要:")
            text_parts.append(summary)
            text_parts.append("")
            text_parts.append("---")
            text_parts.append("")

        # 添加统计信息
        stats = json_data.get("statistics", {})
        text_parts.append("=== 统计信息 ===")
        text_parts.append(f"总关键帧数: {stats.get('total_frames', 0)}")
        text_parts.append(f"有帧的时间段数: {stats.get('segments_with_frames', 0)}")

        return "\n".join(text_parts)

    def _load_notes_knowledge(self, task_id: str) -> Optional[TextKnowledgeBase]:
        """加载指定task_id的笔记知识库"""
        try:
            # 构建笔记文件路径
            notes_path = Path(f"storage/tasks/{task_id}/multimodal_notes/multimodal_notes.json")

            if not notes_path.exists():
                logger.warning(f"笔记文件不存在: {notes_path}")
                return None

            # 检查是否已经加载过
            if task_id in self.knowledge_bases:
                return self.knowledge_bases[task_id]

            # 读取并转换JSON数据
            with open(notes_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 转换为文本格式
            text_content = self._convert_json_to_text(json_data)

            # 创建临时文本文件
            temp_text_path = notes_path.parent / f"temp_notes_{task_id}.txt"
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            # 创建向量数据库
            vector_db = self._get_vector_db(task_id)

            # 创建文本知识库
            knowledge_base = TextKnowledgeBase(
                path=str(temp_text_path),
                vector_db=vector_db,
                embedder=JinaEmbedder(),
            )

            # 加载知识库（首次加载时重建索引）
            knowledge_base.load(recreate=True)

            # 缓存知识库
            self.knowledge_bases[task_id] = knowledge_base

            logger.info(f"成功加载task_id {task_id} 的笔记知识库")
            return knowledge_base

        except Exception as e:
            logger.error(f"加载笔记知识库失败: task_id={task_id}, error={str(e)}")
            return None

    def get_or_create_agent(self, task_id: str, user_id: str = "user") -> Agent:
        """获取或创建指定task_id的agent实例"""
        agent_key = f"{task_id}_{user_id}"

        if agent_key not in self.agent_sessions:
            # 加载笔记知识库
            knowledge_base = self._load_notes_knowledge(task_id)

            # 创建Agent
            agent = Agent(
                name=f"VideoNotes-Agent-{task_id}",
                agent_id=f"video_notes_agent_{task_id}",
                user_id=user_id,
                model=OpenAIChat(id="gpt-4o"),
                knowledge=knowledge_base,
                add_history_to_messages=True,
                num_history_responses=5,
                markdown=True,
                show_tool_calls=True,
                instructions=[
                    "你是一个专业的视频笔记助手。",
                    "你可以基于用户的视频笔记内容回答相关问题。",
                    "如果用户询问的内容在笔记中找不到，请诚实地告知。",
                    "回答时要准确、有条理，并尽可能引用知识库中具体知识。",
                    "使用中文回答，保持友好和专业的语调。"
                ]
            )

            self.agent_sessions[agent_key] = agent
            logger.info(f"为task_id {task_id}, user_id {user_id} 创建新的agent实例")

        return self.agent_sessions[agent_key]

    def run_agent(self, task_id: str, message: str, user_id: str = "user") -> str:
        """运行agent并返回响应"""
        try:
            agent = self.get_or_create_agent(task_id, user_id)
            response = agent.run(message)
            return response.content
        except Exception as e:
            logger.error(f"Agent运行失败: task_id={task_id}, user_id={user_id}, error={str(e)}")
            raise

    def clear_session(self, task_id: str, user_id: str = "user") -> bool:
        """清除指定的agent会话"""
        agent_key = f"{task_id}_{user_id}"
        if agent_key in self.agent_sessions:
            del self.agent_sessions[agent_key]
            logger.info(f"已清除task_id {task_id}, user_id {user_id} 的agent会话")
            return True
        return False

    def reload_knowledge(self, task_id: str) -> bool:
        """重新加载指定task_id的知识库（当笔记更新时调用）"""
        try:
            # 清除缓存的知识库
            if task_id in self.knowledge_bases:
                del self.knowledge_bases[task_id]

            # 清除相关的agent会话
            keys_to_remove = [key for key in self.agent_sessions.keys() if key.startswith(f"{task_id}_")]
            for key in keys_to_remove:
                del self.agent_sessions[key]

            logger.info(f"已重新加载task_id {task_id} 的知识库")
            return True
        except Exception as e:
            logger.error(f"重新加载知识库失败: task_id={task_id}, error={str(e)}")
            return False

# 全局服务实例
video_notes_agent_service = VideoNotesAgentService()
