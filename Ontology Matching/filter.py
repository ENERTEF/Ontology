"""Merge class matching results from name-based and hierarchy-based methods."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from utilizing import get_file_path

# name-based 底集 + hierarchy 加成；weighted = 并集加权后二次阈值
FILTER_MODES = ('boost', 'weighted')
DEFAULT_CLASS_WEIGHTS = (0.7, 0.3)


class Filter(object):
    """Combine name-based and hierarchy class alignments."""

    def process_similarity(self, match_list):
        """
        Group pairs by source entity while preserving flat list output.

        @param {list[tuple]} match_list - (source, target, score) rows
        @returns {list[tuple]} Reordered rows
        """
        grouped_data = defaultdict(list)
        for item in match_list:
            key = item[0]
            grouped_data[key].append(item)
        return [item for sublist in grouped_data.values() for item in sublist]

    @staticmethod
    def _normalize_pairs(match_list):
        """
        @param {list|None} match_list - Raw match rows
        @returns {list[tuple]} Valid (source, target, score) tuples
        """
        if not match_list:
            return []
        return [
            item for item in match_list
            if item is not None and isinstance(item, tuple) and len(item) == 3
        ]

    @staticmethod
    def _pair_score_map(match_list):
        """
        Keep the highest score for each (source, target) pair.

        @param {list[tuple]} match_list - Alignment rows
        @returns {dict[tuple, float]} Pair to similarity
        """
        scores = {}
        for src, tgt, sim in match_list:
            key = (src, tgt)
            value = float(sim)
            if key not in scores or value > scores[key]:
                scores[key] = value
        return scores

    def filter_classes(
        self,
        name_pairs,
        hierarchy_pairs,
        mode='boost',
        weights=DEFAULT_CLASS_WEIGHTS,
        fuse_threshold=None,
    ):
        """
        Merge name-based and hierarchy class pairs.

        Modes:
        - ``boost``: keep only name-based pairs; boost score when hierarchy agrees
        - ``weighted``: union candidates; missing method scores as 0; drop if
          fused score <= fuse_threshold

        @param {list[tuple]} name_pairs - Name-based class alignments
        @param {list[tuple]} hierarchy_pairs - Hierarchy-based class alignments
        @param {str} mode - ``boost`` or ``weighted``
        @param {tuple} weights - (name_weight, hierarchy_weight)
        @param {float|None} fuse_threshold - Required for ``weighted`` mode
        @returns {list[tuple]} Filtered (source, target, fused_score) pairs
        """
        if mode not in FILTER_MODES:
            raise ValueError(f'Unsupported filter mode: {mode}. Use one of {FILTER_MODES}')

        name_pairs = self._normalize_pairs(name_pairs)
        hierarchy_pairs = self._normalize_pairs(hierarchy_pairs)
        w_name, w_hier = weights
        name_scores = self._pair_score_map(name_pairs)
        hier_scores = self._pair_score_map(hierarchy_pairs)

        if mode == 'boost':
            combined = []
            for pair, s_name in name_scores.items():
                s_hier = hier_scores.get(pair)
                if s_hier is None:
                    fused = s_name
                elif s_name == 1.0 or s_hier == 1.0:
                    fused = 1.0
                else:
                    fused = (w_name * s_name + w_hier * s_hier) / (w_name + w_hier)
                combined.append((pair[0], pair[1], round(float(fused), 6)))
            return self.process_similarity(combined)

        # weighted: union + missing=0 + secondary threshold
        if fuse_threshold is None:
            raise ValueError('fuse_threshold is required for weighted mode')

        candidate_pairs = set(name_scores) | set(hier_scores)
        combined = []
        for pair in candidate_pairs:
            s_name = name_scores.get(pair, 0.0)
            s_hier = hier_scores.get(pair, 0.0)
            if s_name == 1.0 or s_hier == 1.0:
                fused = 1.0
            else:
                fused = w_name * s_name + w_hier * s_hier
            if fused > fuse_threshold:
                combined.append((pair[0], pair[1], round(float(fused), 6)))
        return self.process_similarity(combined)

    def get_filter_result(self, *lists, weights=None, mode='boost', fuse_threshold=None):
        """
        Backward-compatible entry: first two lists are name / hierarchy.

        @param {*lists} lists - Match lists (name first, hierarchy second)
        @param {tuple|None} weights - Optional class weights
        @param {str} mode - Filter mode
        @param {float|None} fuse_threshold - Used by weighted mode
        @returns {list[tuple]} Filtered class pairs
        """
        name_pairs = lists[0] if len(lists) > 0 else []
        hierarchy_pairs = lists[1] if len(lists) > 1 else []
        return self.filter_classes(
            name_pairs,
            hierarchy_pairs,
            mode=mode,
            weights=weights or DEFAULT_CLASS_WEIGHTS,
            fuse_threshold=fuse_threshold,
        )

    def save_filter_results(self, filter_list, onto_s_name, onto_t_name, result_folder):
        """
        @param {list[tuple]|None} filter_list - Filtered alignments
        @param {str} onto_s_name - Source ontology name
        @param {str} onto_t_name - Target ontology name
        @param {str} result_folder - Output directory
        @returns {str} Saved Excel path
        """
        if filter_list is None:
            filter_list = []
        if any(isinstance(item, list) for item in filter_list):
            data_list = [it for sub in filter_list for it in sub]
        else:
            data_list = list(filter_list)

        if len(data_list) > 0:
            ok = all(isinstance(item, tuple) and len(item) == 3 for item in data_list)
            if not ok:
                raise ValueError('Invalid data format: expected iterable of (source, target, score)')

        if len(data_list) == 0:
            df = pd.DataFrame(columns=['Source', 'Target', 'Cosine Similarity'])
        else:
            df = pd.DataFrame(data_list, columns=['Source', 'Target', 'Cosine Similarity'])

        out_dir = Path(result_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = '_filter_results.xlsx'
        file_name = onto_s_name + '_' + onto_t_name
        try:
            output_path = get_file_path(str(out_dir), file_name, suffix)
        except Exception:
            output_path = str(out_dir / f'{file_name}{suffix}')

        df.to_excel(output_path, index=False, engine='openpyxl')
        return output_path


if __name__ == '__main__':
    result_filter = Filter()
    list1 = [
        ('test sample', 'visit', 0.75376976),
        ('test sample', 'role', 0.7),
        ('storage', 'building', 0.7028562),
    ]
    list2 = [
        ('test sample', 'visit', 1),
        ('test', 'space', 0.8),
        ('test sample', 'role', 0.8),
        ('storage', 'building', 0.7),
    ]

    print('boost:', result_filter.filter_classes(list1, list2, mode='boost'))
    print(
        'weighted:',
        result_filter.filter_classes(list1, list2, mode='weighted', fuse_threshold=0.5),
    )
