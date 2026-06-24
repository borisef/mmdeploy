from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.heatmap_heads.atraf.'
    'heatmap_head_with_classifiers.HeatmapHeadWithClassifiers.forward')
def heatmap_head_with_classifiers__forward(self, feats):
    ctx = FUNCTION_REWRITER.get_context()
    heatmaps = ctx.origin_func(self, feats)
    outputs = [heatmaps]
    for head in self.classifier_heads:
        outputs.append(head(feats[0]))
    return tuple(outputs)
