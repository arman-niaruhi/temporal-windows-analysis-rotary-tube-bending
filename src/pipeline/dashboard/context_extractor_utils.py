import os
import streamlit as st
from mlflow.tracking import MlflowClient
MLFLOW_TRACKING_URI = "mlruns"  

def get_experiment_runs():
        """Get all runs for a given experiment ID using MLflow, optionally filtered by search term"""
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            experiment_id = "665463947744551178"
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"]  
            )
            
            run_info = []
            for run in runs:
                run_name = run.data.tags.get("mlflow.runName", "")
                
                if not run_name:
                    run_name = run.info.run_id
                
                run_id = run.info.run_id
                
                run_info.append({
                        'name': run_name,
                        'id': run_id,
                        'display': f"{run_name} ({run_id[:8]})",
                        'start_time': run.info.start_time
                    })
            
            return run_info
            
        except Exception as e:
            st.error(f"Error fetching runs from MLflow: {e}")
            import traceback
            st.error(traceback.format_exc())
            return get_experiment_runs_fallback()


def get_experiment_runs_fallback():
    """Fallback method: Get runs by listing directories"""
    runs = []
    experiment_path = os.path.join(MLFLOW_TRACKING_URI, "665463947744551178")
    
    if os.path.exists(experiment_path):
        all_runs = [d for d in os.listdir(experiment_path)
                    if os.path.isdir(os.path.join(experiment_path, d))]
        
        filtered = all_runs
        runs = [{'name': r, 'id': r, 'display': r, 'start_time': None} for r in sorted(filtered)]
    
    return runs


def get_all_run_names():
    """Get a list of all unique run names from the experiment"""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment_id = "665463947744551178"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )
        
        run_names = []
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if run_name and run_name not in run_names:
                run_names.append(run_name)
        
        return sorted(run_names)
        
    except Exception as e:
        st.error(f"Error fetching run names: {e}")
        return []


def find_png_images_with_rules(root_path):
    """Find PNG images with rules: 
    - For special folders, take only last epoch
    - For attention_lines, collect all angle images"""
    import glob
    from pathlib import Path
    import re
    
    all_images = []
    attention_line_images = {}  
    
    root_path = Path(root_path)
    
    pattern = str(root_path / "**" / "*.png")
    all_pngs = glob.glob(pattern, recursive=True)
    
    images_by_dir = {}
    for png_path in all_pngs:
        dir_path = os.path.dirname(png_path)
        if dir_path not in images_by_dir:
            images_by_dir[dir_path] = []
        images_by_dir[dir_path].append(png_path)
    
    for dir_path, images in images_by_dir.items():
        dir_name = os.path.basename(dir_path)
        
        if "attention_lines" in dir_name.lower() or "04_attention_lines" in dir_name:
            for img_path in images:
                img_name = os.path.basename(img_path)
                
                match = re.search(r'attention_angle_(\d+)\.png$', img_name, re.IGNORECASE)
                if match:
                    angle_num = int(match.group(1))
                    attention_line_images[angle_num] = img_path
                else:
                    all_images.append(img_path)
        
        elif any(special in dir_name.lower() 
                for special in ["predictions", "loss", "attention"]):
            if "attention_lines" not in dir_name.lower():
                epoch_images = {}
                
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    
                    match = re.search(r'_epoch_(\d+)\.png$', img_name, re.IGNORECASE)
                    if match:
                        epoch_num = int(match.group(1))
                        epoch_images[epoch_num] = img_path
                    else:
                        all_images.append(img_path)
                
                if epoch_images:
                    max_epoch = max(epoch_images.keys())
                    all_images.append(epoch_images[max_epoch])
                else:
                    all_images.extend(images)
        else:
            all_images.extend(images)
    
    return list(set(all_images)), attention_line_images

    
def create_video_from_images(image_paths, output_dir, fps=5):
    """Create a video from a list of image paths with better codec and looping"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import tempfile
        
        if not image_paths:
            st.error("No image paths provided")
            return None
        
        st.info(f"Creating video from {len(image_paths)} images at {fps} FPS...")
        
        frames = []
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    pil_img = Image.open(img_path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                frames.append(img)
            except Exception as e:
                st.warning(f"Failed to load image {img_path}: {e}")
                continue
        
        if not frames:
            st.error("No frames could be loaded")
            return None
        
        height, width = frames[0].shape[:2]
        
        resized_frames = []
        for img in frames:
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            resized_frames.append(img)
        
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=output_dir)
        temp_video_path = temp_video.name
        temp_video.close()
        
        codecs_to_try = [
            ('mp4v', 'MPEG-4')
        ]
        
        video = None
        for codec_code, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                
                if video.isOpened():
                    st.success(f"Using {codec_name} codec")
                    break
                else:
                    video.release()
                    video = None
            except:
                continue
        
        if video is None or not video.isOpened():
            st.error("Failed to create video with any codec")
            return None
        
        num_loops = 3  
        
        for loop in range(num_loops):
            for frame in resized_frames:
                video.write(frame)
            
            if loop < num_loops - 1:
                for _ in range(fps // 2):  
                    video.write(resized_frames[-1])
        
        for _ in range(fps * 2):  
            video.write(resized_frames[-1])
        
        video.release()
        
        if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
            st.success(f"Video created successfully: {os.path.getsize(temp_video_path)} bytes")
            return temp_video_path
        else:
            st.error("Video file is empty or wasn't created")
            return None
            
    except Exception as e:
        st.error(f"Error creating video: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None