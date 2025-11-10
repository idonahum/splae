class BaseAccuracyEstimator:
    @staticmethod
    def _zip_results(real_scores, pred_scores, img_name, certainty_score=None, reliability_score=None, forground_ratio=None):
        results_by_metric = {
            metric: {
                'real_score': real_scores[metric].cpu().numpy().tolist(),
                'pred_score': pred_scores[metric].cpu().numpy().tolist(),
                'gaps': abs(real_scores[metric].cpu().numpy() - pred_scores[metric].cpu().numpy()).tolist()
            }
            for metric in real_scores.keys()
        }
        return {'img_name': img_name, 'results_by_metric': results_by_metric, 'certainty_score': certainty_score, 'reliability_score': reliability_score, 'forground_ratio': forground_ratio}