"""多模态服务 - 统一的帧提取、嵌入生成和去重服务"""
import os,base64,time,logging,tempfile,shutil,json,concurrent.futures,threading,requests
from typing import List,Optional,Dict,Any,Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
from services.ffmpeg_process import VideoProcessor

class MultimodalService:
    """多模态服务 - 统一处理帧提取、嵌入生成和去重"""
    EMBED_MODEL="jina-embeddings-v4"
    BATCH_SIZE=10
    API_DELAY=0.1
    JINA_API_URL="https://api.jina.ai/v1/embeddings"
    IMG_FORMATS={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.webp':'image/webp'}

    def __init__(self,jina_api_key:str,ffmpeg_path:str="ffmpeg",similarity_threshold:float=0.9,
                 embedding_model:str=EMBED_MODEL,batch_size:int=BATCH_SIZE,frame_fps:float=0.2,
                 max_concurrent_segments:int=3,enable_text_alignment:bool=True,max_aligned_frames:int=3,
                 logger:Optional[logging.Logger]=None):
        """初始化多模态服务"""
        self.api_key=jina_api_key
        self.sim_thresh=similarity_threshold
        self.embed_model=embedding_model
        self.batch_sz=batch_size
        self.fps=frame_fps
        self.max_workers=max_concurrent_segments
        self.enable_text_alignment=enable_text_alignment
        self.max_aligned_frames=max_aligned_frames
        self.api_url=self.JINA_API_URL
        self.video_proc=VideoProcessor(ffmpeg_path)
        self.logger=logger or logging.getLogger(__name__)
        self._lock=threading.Lock()

    def extract_frames(self,video_path:str,start_time:float,end_time:float,fps:float=1.0,output_dir:Optional[str]=None)->List[str]:
        """从视频指定时间范围抽取帧"""
        return self.video_proc.extract_frames(video_path,start_time,end_time,fps,output_dir)

    def _get_mime_type(self,path:str)->str:
        """检测图片MIME类型"""
        p=path.lower()
        for ext,mime in self.IMG_FORMATS.items():
            if p.endswith(ext):return mime
        return "image/jpeg"

    def _to_base64(self,path:str)->str:
        """转换图片为base64格式（Jina AI 需要纯 base64，不带 data: 前缀）"""
        with open(path,"rb") as f:
            data=base64.b64encode(f.read()).decode('utf-8')
            return data

    def generate_embeddings(self,paths:List[str],batch_size:Optional[int]=None,lock:Optional[object]=None)->Tuple[List[np.ndarray],List[str]]:
        """批量获取图片embeddings"""
        bs=batch_size or self.batch_sz
        embeds,success=[],[]

        for i in range(0,len(paths),bs):
            batch=paths[i:i+bs]
            self.logger.info(f"Processing batch {i//bs+1}/{(len(paths)+bs-1)//bs}")

            data,valid=[],[]
            for p in batch:
                try:
                    b64=self._to_base64(p)
                    data.append({"image":b64})
                    valid.append(p)
                except Exception as e:
                    self.logger.warning(f"Failed to process {p}: {e}")

            if not data:continue

            try:
                if lock:
                    with lock:
                        emb=self._call_api(data)
                        time.sleep(self.API_DELAY)
                else:
                    emb=self._call_api(data)
                    time.sleep(self.API_DELAY)
                embeds.extend(emb)
                success.extend(valid)
            except Exception as e:
                self.logger.warning(f"API call failed: {e}")

        return embeds,success

    def _call_api(self,data:List[Dict[str,Any]])->List[np.ndarray]:
        """调用 Jina AI API 获取 embeddings"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.embed_model,
                "task": "text-matching",
                "input": data
            }

            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Jina AI 返回格式: {"data": [{"embedding": [...]}, ...]}
            if "data" not in result:
                raise RuntimeError(f"Unexpected Jina AI response format: {result}")

            embeddings = []
            for item in result["data"]:
                if "embedding" in item:
                    embeddings.append(np.array(item["embedding"]))
                else:
                    raise RuntimeError(f"Missing embedding in response item: {item}")

            return embeddings

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Jina AI API request error: {e}")
        except Exception as e:
            raise RuntimeError(f"Jina AI API error: {e}")

    @staticmethod
    def calc_similarity(e1:np.ndarray,e2:np.ndarray)->float:
        """计算余弦相似度"""
        n1,n2=np.linalg.norm(e1),np.linalg.norm(e2)
        return 0.0 if n1==0 or n2==0 else float(np.dot(e1,e2)/(n1*n2))

    def remove_duplicates(self,paths:List[str],embeds:List[np.ndarray])->List[str]:
        """基于embeddings相似度去除重复图片"""
        if len(paths)!=len(embeds):raise ValueError("paths and embeddings length mismatch")

        unique_idx,processed=[],[]
        for i,curr in enumerate(embeds):
            is_dup=False
            for proc in processed:
                if self.calc_similarity(curr,proc)>self.sim_thresh:
                    is_dup=True
                    self.logger.debug(f"Duplicate: {paths[i]}")
                    break
            if not is_dup:
                unique_idx.append(i)
                processed.append(curr)

        result=[paths[i] for i in unique_idx]
        self.logger.info(f"Removed {len(paths)-len(result)} duplicates, kept {len(result)}")
        return result

    def save_unique_frames(self,paths:List[str],output_dir:str,copy_files:bool=True)->List[str]:
        """保存去重后的图片"""
        os.makedirs(output_dir,exist_ok=True)
        saved=[]
        for i,src in enumerate(paths):
            dst=os.path.join(output_dir,f"unique_frame_{i+1:06d}.jpg")
            try:
                (shutil.copy2 if copy_files else shutil.move)(src,dst)
                saved.append(dst)
            except Exception as e:
                self.logger.warning(f"Failed to save {src}: {e}")
        self.logger.info(f"Saved {len(saved)} frames to {output_dir}")
        return saved

    def process_video_frames(self,video_path:str,start_time:float,end_time:float,output_dir:str,
                           fps:float=1.0,temp_dir:Optional[str]=None,keep_temp:bool=False,
                           lock:Optional[object]=None)->Dict[str,Any]:
        """视频帧去重处理流程"""
        self.logger.info(f"Processing {video_path} [{start_time}s-{end_time}s] fps={fps} thresh={self.sim_thresh}")

        temp_created=temp_dir is None
        if temp_created:temp_dir=tempfile.mkdtemp(prefix="video_dedup_")
        else:os.makedirs(temp_dir,exist_ok=True)

        try:
            self.logger.info("1. Extracting frames...")
            frames=self.extract_frames(video_path,start_time,end_time,fps,temp_dir)

            self.logger.info("2. Getting embeddings...")
            embeds,success=self.generate_embeddings(frames,lock=lock)
            if len(embeds)!=len(frames):
                self.logger.warning(f"Got {len(embeds)} embeddings for {len(frames)} frames")
                frames=success

            self.logger.info("3. Removing duplicates...")
            unique=self.remove_duplicates(frames,embeds)

            self.logger.info("4. Saving unique frames...")
            saved=self.save_unique_frames(unique,output_dir)

            result={"video_path":video_path,"time_range":(start_time,end_time),"fps":fps,
                   "total_frames":len(frames),"unique_frames":len(saved),
                   "duplicates_removed":len(frames)-len(saved),"similarity_threshold":self.sim_thresh,
                   "output_dir":output_dir,"saved_paths":saved}

            self.logger.info("✅ Process completed!")
            return result
        finally:
            if temp_created and not keep_temp:
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned temp: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed cleanup: {e}")

    def _parse_time(self,t:str)->float:
        """时间字符串转秒数"""
        try:
            parts=t.split(':')
            h,m=int(parts[0]),int(parts[1])
            s_parts=parts[2].split('.')
            s=int(s_parts[0])
            ms=int(s_parts[1]) if len(s_parts)>1 else 0
            return h*3600+m*60+s+ms/1000
        except (ValueError,IndexError) as e:
            raise ValueError(f"时间格式错误: {t}, {e}")

    def extract_segment_frames(self,video_path:str,start_time:str,end_time:str,output_dir:str,
                             text_summary:str="",enable_alignment:bool=True)->List[str]:
        """提取时间段视频帧并去重，支持图文对齐"""
        start_sec,end_sec=self._parse_time(start_time),self._parse_time(end_time)
        seg_dir=os.path.join(output_dir,f"segment_{start_time.replace(':','-')}_to_{end_time.replace(':','-')}")
        os.makedirs(seg_dir,exist_ok=True)

        try:
            # 1. 先提取和去重帧
            result=self.process_video_frames(video_path,start_sec,end_sec,seg_dir,
                                           self.fps,keep_temp=False,lock=self._lock)
            frame_paths=result.get("saved_paths",[])

            # 2. 如果启用对齐且有文本摘要，进行图文对齐
            if enable_alignment and self.enable_text_alignment and text_summary.strip() and frame_paths:
                aligned_paths=self.align_frames_with_text(frame_paths,text_summary)
                # 重新保存对齐后的帧到新目录
                aligned_dir=os.path.join(seg_dir,"aligned")
                final_paths=self.save_unique_frames(aligned_paths,aligned_dir,copy_files=True)
                return final_paths

            return frame_paths
        except Exception as e:
            self.logger.error(f"段 {start_time}-{end_time} 处理失败: {e}")
            return []

    def _process_segment(self,data:tuple)->dict:
        """处理单个时间段"""
        i,seg,vid_path,frames_dir,out_dir=data
        start,end,summary=seg.get("start_time",""),seg.get("end_time",""),seg.get("summary","")

        try:
            # 传入摘要文本进行图文对齐
            paths=self.extract_segment_frames(vid_path,start,end,frames_dir,summary,enable_alignment=True)
            rel_paths=[os.path.relpath(p,out_dir) for p in paths]
        except Exception as e:
            self.logger.error(f"段 {start}-{end} 失败: {e}")
            rel_paths=[]

        return {"segment_id":i+1,"start_time":start,"end_time":end,
                "duration_seconds":self._parse_time(end)-self._parse_time(start),
                "summary":summary,"key_frames":rel_paths,"frame_count":len(rel_paths)}

    def align_frames_with_text(self,frame_paths:List[str],text_summary:str,max_frames:int=None)->List[str]:
        """使用Cohere跨模态嵌入对齐图文"""
        if max_frames is None:max_frames=self.max_aligned_frames
        if not frame_paths or not text_summary.strip():
            return frame_paths[:max_frames]

        try:
            # 1. 获取文本嵌入
            text_input=[{"text":text_summary}]
            text_embed=self._call_api(text_input)[0]

            # 2. 获取图像嵌入
            img_embeds,valid_paths=self.generate_embeddings(frame_paths)
            if not img_embeds:
                return frame_paths[:max_frames]

            # 3. 计算跨模态相似度
            similarities=[]
            for img_embed in img_embeds:
                sim=self.calc_similarity(text_embed,img_embed)
                similarities.append(sim)

            # 4. 按相似度排序，选择最匹配的帧
            sorted_indices=sorted(range(len(similarities)),key=lambda i:similarities[i],reverse=True)
            aligned_paths=[valid_paths[i] for i in sorted_indices[:max_frames]]

            self.logger.info(f"文本对齐: {len(frame_paths)}帧 -> {len(aligned_paths)}帧, 最高相似度: {max(similarities):.3f}")
            return aligned_paths

        except Exception as e:
            self.logger.warning(f"图文对齐失败: {e}, 使用原始帧")
            return frame_paths[:max_frames]

    def generate_multimodal_notes(self,video_path:str,summary_json_path:str,output_dir:str)->str:
        """生成图文混排笔记（并发处理）"""
        with open(summary_json_path,'r',encoding='utf-8') as f:
            data=json.load(f)

        summaries=data.get("summaries",[])
        if not summaries:raise ValueError("摘要数据为空")

        os.makedirs(output_dir,exist_ok=True)
        frames_dir=os.path.join(output_dir,"frames")
        os.makedirs(frames_dir,exist_ok=True)

        tasks=[(i,seg,video_path,frames_dir,output_dir) for i,seg in enumerate(summaries)]
        notes=[]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx={executor.submit(self._process_segment,task):task[0] for task in tasks}
            results={}

            for future in concurrent.futures.as_completed(future_to_idx):
                idx=future_to_idx[future]
                try:
                    results[idx]=future.result()
                except Exception as e:
                    self.logger.error(f"段 {idx+1} 失败: {e}")
                    seg=summaries[idx]
                    results[idx]={"segment_id":idx+1,"start_time":seg.get("start_time",""),
                                 "end_time":seg.get("end_time",""),"duration_seconds":0,
                                 "summary":seg.get("summary",""),"key_frames":[],"frame_count":0}

            for i in range(len(summaries)):
                if i in results:notes.append(results[i])

        final={"video_info":{"source_video":os.path.basename(video_path),"total_segments":len(notes),
                            "generated_at":datetime.now().isoformat(),
                            "processing_mode":f"concurrent (workers={self.max_workers})"},
               "segments":notes,
               "statistics":{"total_frames":sum(n["frame_count"] for n in notes),
                           "segments_with_frames":len([n for n in notes if n["frame_count"]>0])}}

        out_file=os.path.join(output_dir,"multimodal_notes.json")
        with open(out_file,'w',encoding='utf-8') as f:
            json.dump(final,f,ensure_ascii=False,indent=4)
        return out_file

    def export_to_markdown(self,notes_json_path:str,output_path:str=None,image_base_path:str=None)->str:
        """导出为Markdown格式"""
        with open(notes_json_path,'r',encoding='utf-8') as f:
            data=json.load(f)

        if not output_path:output_path=f"{Path(notes_json_path).stem}.md"
        if not image_base_path:image_base_path=str(Path(notes_json_path).parent)

        content=self._gen_markdown(data,output_path,image_base_path)
        with open(output_path,'w',encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _gen_markdown(self,data:Dict[str,Any],output_path:str=None,img_base:str=None)->str:
        """生成Markdown内容"""
        info,segs,stats=data.get("video_info",{}),data.get("segments",[]),data.get("statistics",{})
        lines=[]

        # 标题和基本信息
        lines.extend([f"# 📹 视频笔记：{info.get('source_video','未知视频')}","",
                     "## 📊 基本信息","",
                     f"- **视频文件**: {info.get('source_video','未知')}",
                     f"- **生成时间**: {info.get('generated_at','未知')}",
                     f"- **总时间段**: {info.get('total_segments',0)}",
                     f"- **总关键帧**: {stats.get('total_frames',0)}",
                     f"- **有效时间段**: {stats.get('segments_with_frames',0)}",""])

        # 目录
        lines.extend(["## 📑 目录",""])
        for i,seg in enumerate(segs,1):
            start,end=seg.get("start_time",""),seg.get("end_time","")
            lines.append(f"{i}. [{start} - {end}](#section-{i})")
        lines.extend(["","## 📝 详细内容",""])

        # 详细内容
        for i,seg in enumerate(segs,1):
            start,end=seg.get("start_time",""),seg.get("end_time","")
            dur,summary=seg.get("duration_seconds",0),seg.get("summary","")
            frames=seg.get("key_frames",[])

            lines.extend([f"### <a id='section-{i}'></a>时间段 {i}","",f"**⏰ 时间**: {start} - {end} ({dur:.1f}秒)","",
                         "**📋 摘要**:","",summary,""])

            if frames:
                lines.extend([f"**🖼️ 关键帧** ({len(frames)}张):",""])
                for fp in frames:
                    name=Path(fp).name
                    path=f"/{img_base}/multimodal_notes/{fp}" if img_base else f"multimodal_notes/{fp}"
                    lines.append(f"![{name}]({path})")
                lines.append("")
            else:
                lines.extend(["*该时间段无关键帧*",""])
            lines.extend(["---",""])

        # 页脚
        lines.extend(["## 🔧 生成信息","","本笔记由视频处理 API 自动生成",
                     f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        if output_path:lines.append(f"输出文件: {output_path}")

        return "\n".join(lines)


def create_multimodal_service(jina_api_key:str,enable_text_alignment:bool=True,**kwargs)->MultimodalService:
    """便捷函数：创建多模态服务"""
    return MultimodalService(jina_api_key,enable_text_alignment=enable_text_alignment,**kwargs)
