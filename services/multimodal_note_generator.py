#!/usr/bin/env python3
"""
图文混排笔记生成器 - 门面类，委托给MultimodalService处理
"""
import logging
from typing import Optional
from services.multimodal_service import MultimodalService
from settings import get_settings
class MultimodalNoteGenerator:
    def __init__(self, logger: Optional[logging.Logger] = None):

        s = get_settings()
        key = s.JINA_API_KEY
        if not key:
            raise ValueError("JINA_API_KEY 不能为空")

        self.logger = logger or logging.getLogger(__name__)
        try:
            self.multimodal_service = MultimodalService(
                jina_api_key=key,
                ffmpeg_path=s.FFMPEG_PATH,
                similarity_threshold=s.MULTIMODAL_SIMILARITY_THRESHOLD,
                frame_fps=s.MULTIMODAL_FRAME_FPS,
                max_concurrent_segments=s.MULTIMODAL_MAX_CONCURRENT_SEGMENTS,
                enable_text_alignment=s.MULTIMODAL_ENABLE_TEXT_ALIGNMENT,
                max_aligned_frames=s.MULTIMODAL_MAX_ALIGNED_FRAMES,
                logger=self.logger
            )
            self.multimodal_service.embed_model = s.MULTIMODAL_EMBED_MODEL
            self.multimodal_service.batch_sz = s.MULTIMODAL_BATCH_SIZE
            self.multimodal_service.API_DELAY = s.MULTIMODAL_API_DELAY
        except Exception as e:
            self.logger.error(f"MultimodalService 初始化失败: {e}")
            raise

    def generate_multimodal_notes(self, video_path: str, summary_json_path: str, output_dir: str) -> str:
        """生成图文混排笔记 - 委托给MultimodalService"""
        return self.multimodal_service.generate_multimodal_notes(video_path, summary_json_path, output_dir)

    def export_to_markdown(self, notes_json_path: str, output_path: str = None,
                          image_base_path: str = None) -> str:
        """导出Markdown格式 - 委托给MultimodalService"""
        return self.multimodal_service.export_to_markdown(notes_json_path, output_path, image_base_path)



