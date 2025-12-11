import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
from numba import jit

warnings.filterwarnings('ignore')


class InstanceSegmentationEvaluator:
    """
    Evaluates instance segmentation predictions using mean Average Precision (mAP).
    Works with both 2D and 3D TIF files where each instance has a unique value.
    """
    
    def __init__(self, iou_thresholds: np.ndarray = None):
        """
        Args:
            iou_thresholds: Array of IoU thresholds for mAP calculation
                           Default: [0.5, 0.55, 0.6, ..., 0.95] (COCO-style)
        """
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = iou_thresholds
    
    def load_mask(self, filepath: str) -> np.ndarray:
        """Load TIF mask file."""
        return io.imread(filepath)
    
    def extract_instances(self, mask: np.ndarray) -> List[int]:
        """
        Extract instance IDs from a labeled mask.
        
        Args:
            mask: Labeled mask where each instance has unique value
            
        Returns:
            List of instance IDs (excluding background)
        """
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # Remove background
        return instance_ids.tolist()
    
    def compute_iou_fast(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                         pred_id: int, gt_id: int) -> float:
        """
        Compute IoU between two instances in labeled masks (optimized).
        
        Args:
            pred_mask: Prediction labeled mask
            gt_mask: Ground truth labeled mask
            pred_id: Instance ID in prediction mask
            gt_id: Instance ID in ground truth mask
            
        Returns:
            IoU score
        """
        pred_binary = (pred_mask == pred_id)
        gt_binary = (gt_mask == gt_id)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        if intersection == 0:
            return 0.0
        
        union = np.logical_or(pred_binary, gt_binary).sum()
        return float(intersection) / float(union)
    
    def match_instances(self, pred_mask: np.ndarray, gt_mask: np.ndarray,
                       pred_ids: List[int], gt_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predicted instances to ground truth using Hungarian algorithm (optimized).
        
        Args:
            pred_mask: Prediction labeled mask
            gt_mask: Ground truth labeled mask
            pred_ids: List of prediction instance IDs
            gt_ids: List of ground truth instance IDs
            
        Returns:
            iou_matrix: IoU scores between all pred-gt pairs
            matches: Tuple of (pred_indices, gt_indices) for matched pairs
        """
        n_pred = len(pred_ids)
        n_gt = len(gt_ids)
        
        if n_pred == 0 or n_gt == 0:
            return np.zeros((n_pred, n_gt)), (np.array([]), np.array([]))
        
        # Compute IoU matrix (optimized)
        iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i, pred_id in enumerate(pred_ids):
            for j, gt_id in enumerate(gt_ids):
                iou_matrix[i, j] = self.compute_iou_fast(pred_mask, gt_mask, pred_id, gt_id)
        
        # Use Hungarian algorithm for optimal matching
        # Maximize IoU by minimizing negative IoU
        pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
        
        return iou_matrix, (pred_idx, gt_idx)
    
    def compute_ap_at_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray,
                          pred_ids: List[int], gt_ids: List[int],
                          iou_threshold: float) -> float:
        """
        Compute Average Precision at a specific IoU threshold.
        
        Args:
            pred_mask: Prediction labeled mask
            gt_mask: Ground truth labeled mask
            pred_ids: List of prediction instance IDs
            gt_ids: List of ground truth instance IDs
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Average Precision score
        """
        n_gt = len(gt_ids)
        
        if n_gt == 0:
            return 1.0 if len(pred_ids) == 0 else 0.0
        
        if len(pred_ids) == 0:
            return 0.0
        
        # Match instances
        iou_matrix, (pred_idx, gt_idx) = self.match_instances(pred_mask, gt_mask, pred_ids, gt_ids)
        
        # Count true positives and false positives
        tp = 0
        matched_gt = set()
        
        for p_idx, g_idx in zip(pred_idx, gt_idx):
            if iou_matrix[p_idx, g_idx] >= iou_threshold:
                tp += 1
                matched_gt.add(g_idx)
        
        fp = len(pred_ids) - tp
        fn = n_gt - len(matched_gt)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # For single image, AP = precision at max recall
        return precision
    
    def compute_map(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
        """
        Compute mean Average Precision across all IoU thresholds (optimized).
        
        Args:
            pred_mask: Predicted instance mask (labeled)
            gt_mask: Ground truth instance mask (labeled)
            
        Returns:
            Dictionary with mAP and AP at each threshold
        """
        pred_ids = self.extract_instances(pred_mask)
        gt_ids = self.extract_instances(gt_mask)
        
        ap_scores = []
        ap_dict = {}
        
        for iou_thresh in self.iou_thresholds:
            ap = self.compute_ap_at_iou(pred_mask, gt_mask, pred_ids, gt_ids, iou_thresh)
            ap_scores.append(ap)
            ap_dict[f'AP@{iou_thresh:.2f}'] = ap
        
        results = {
            'mAP': np.mean(ap_scores),
            'AP50': ap_dict.get('AP@0.50', 0.0),
            'AP75': ap_dict.get('AP@0.75', 0.0),
            'n_pred': len(pred_ids),
            'n_gt': len(gt_ids),
            **ap_dict
        }
        
        return results
    
    def evaluate_files(self, pred_path: str, gt_path: str) -> Dict:
        """
        Evaluate prediction file against ground truth file.
        
        Args:
            pred_path: Path to prediction TIF file
            gt_path: Path to ground truth TIF file
            
        Returns:
            Dictionary with evaluation metrics
        """
        pred_mask = self.load_mask(pred_path)
        gt_mask = self.load_mask(gt_path)
        
        return self.compute_map(pred_mask, gt_mask)
    
    def evaluate_batch(self, pred_dir: str, gt_dir: str, 
                       file_pattern: str = "*.tif") -> Dict:
        """
        Evaluate multiple files in directories.
        
        Args:
            pred_dir: Directory containing prediction TIF files
            gt_dir: Directory containing ground truth TIF files
            file_pattern: Pattern to match files (default: "*.tif")
            
        Returns:
            Dictionary with aggregated metrics
        """
        pred_path = Path(pred_dir)
        gt_path = Path(gt_dir)
        
        pred_files = sorted(pred_path.glob(file_pattern))
        
        all_results = []
        
        for pred_file in pred_files:
            gt_file = gt_path / pred_file.name
            
            if not gt_file.exists():
                print(f"Warning: No ground truth found for {pred_file.name}")
                continue
            
            results = self.evaluate_files(str(pred_file), str(gt_file))
            results['filename'] = pred_file.name
            all_results.append(results)
        
        # Aggregate results
        if not all_results:
            return {}
        
        aggregated = {
            'mAP': np.mean([r['mAP'] for r in all_results]),
            'AP50': np.mean([r['AP50'] for r in all_results]),
            'AP75': np.mean([r['AP75'] for r in all_results]),
            'total_pred': sum([r['n_pred'] for r in all_results]),
            'total_gt': sum([r['n_gt'] for r in all_results]),
            'n_images': len(all_results),
            'per_image_results': all_results
        }
        
        return aggregated


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = InstanceSegmentationEvaluator()
    
    # Single file evaluation
    print("Single file evaluation:")
    results = evaluator.evaluate_files('prediction.tif', 'groundtruth.tif')
    print(f"mAP: {results['mAP']:.4f}")
    print(f"AP@0.50: {results['AP50']:.4f}")
    print(f"AP@0.75: {results['AP75']:.4f}")
    print(f"Predicted instances: {results['n_pred']}")
    print(f"Ground truth instances: {results['n_gt']}")
    
    # Batch evaluation
    print("\nBatch evaluation:")
    batch_results = evaluator.evaluate_batch('predictions/', 'groundtruths/')
    print(f"Overall mAP: {batch_results['mAP']:.4f}")
    print(f"Overall AP@0.50: {batch_results['AP50']:.4f}")
    print(f"Total images: {batch_results['n_images']}")
    
    # Custom IoU thresholds
    print("\nCustom IoU thresholds:")
    custom_evaluator = InstanceSegmentationEvaluator(
        iou_thresholds=np.array([0.5, 0.7, 0.9])
    )
    custom_results = custom_evaluator.evaluate_files('prediction.tif', 'groundtruth.tif')
    print(f"mAP (0.5, 0.7, 0.9): {custom_results['mAP']:.4f}")