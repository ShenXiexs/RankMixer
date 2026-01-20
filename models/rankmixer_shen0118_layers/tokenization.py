import re
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


DEFAULT_SEMANTIC_GROUP_RULES = [
    {"name": "core_id", "patterns": [r"^combination_un_id$"]},
    {"name": "seq", "patterns": [r"^seq::", r"^seq_"]},
    {"name": "dpa", "patterns": [r"^dpa_"]},
    {"name": "item_meta", "patterns": [
        r"^brand_name$", r"^first_category$", r"^second_category$", r"^annual_vol$",
        r"^shop_id$", r"^shop_name$", r"^shop_source$"
    ]},
    {"name": "price", "patterns": [
        r"^reserve_price$", r"^final_promotion_price$", r"^commission$", r"^commission_rate$"
    ]},
    {"name": "semantics", "patterns": [r"^title_sem_id$", r"^image_sem_id$"]},
    {"name": "adslot", "patterns": [
        r"^adx_adslot_id$", r"^ssp_adslot_id$", r"^adslot_id$", r"^channel_id$",
        r"^adslot_id_type$", r"^source_adslot_type$", r"^bid_floor$",
        r"^ad_idea_id$", r"^ad_unit_id$", r"^template_id$", r"^template_type$",
        r"^promotion_type$", r"^target_type$"
    ]},
    {"name": "app", "patterns": [
        r"^app_pkg_src$", r"^app_pkg$", r"^app_src_", r"^package_name$", r"^app_first_type$", r"^app_second_type$"
    ]},
    {"name": "device", "patterns": [
        r"^device_", r"^network$", r"^ip_region$", r"^ip_city$", r"^device_size$", r"^city_level$"
    ]},
    {"name": "strategy", "patterns": [
        r"^model_type$", r"^dispatch_center_id$", r"^rta_type$", r"^crowd_type$", r"^is_new_item$"
    ]},
    {"name": "time", "patterns": [r"^day_h$"]},
    {"name": "user_stat", "patterns": [r"^user__"]},
    {"name": "item_stat", "patterns": [r"^item__"]},
    {"name": "skuid_key_one", "patterns": [r"^skuid__key_one__"]},
    {"name": "skuid_key_two", "patterns": [r"^skuid__key_two__"]},
    {"name": "skuid_key_three", "patterns": [r"^skuid__key_three__"]},
    {"name": "skuid_key_four", "patterns": [r"^skuid__key_four__"]},
    {"name": "skuid_key_five", "patterns": [r"^skuid__key_five__"]},
    {"name": "skuid_stat", "patterns": [r"^skuid__"]},
    {"name": "tsd_stat", "patterns": [r"^tsd__"]},
    {"name": "isd_stat", "patterns": [r"^isd__"]},
]


def _sanitize_group_name(name):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "group"


def _looks_like_regex(pattern):
    if pattern.startswith("re:"):
        return True
    for token in ("^", "$", ".*", "[", "]", "(", ")", "|", "?"):
        if token in pattern:
            return True
    return False


def _normalize_groups(semantic_groups):
    if not semantic_groups:
        return []
    if isinstance(semantic_groups, dict):
        return [(str(k), list(v)) for k, v in semantic_groups.items()]
    if isinstance(semantic_groups, (list, tuple)):
        groups = []
        for idx, item in enumerate(semantic_groups):
            if isinstance(item, dict):
                name = item.get("name", "group_%d" % idx)
                feats = item.get("features") or item.get("patterns") or []
                groups.append((str(name), list(feats)))
            elif isinstance(item, (list, tuple)):
                groups.append(("group_%d" % idx, list(item)))
        return groups
    return []


def _compile_group_rules(group_rules):
    rules = group_rules or DEFAULT_SEMANTIC_GROUP_RULES
    compiled = []
    for rule in rules:
        name = _sanitize_group_name(rule.get("name", "group"))
        patterns = [p for p in rule.get("patterns", []) if p]
        if not patterns:
            continue
        compiled.append((name, [re.compile(p) for p in patterns]))
    return compiled


def _assign_semantic_groups(feature_names, group_rules):
    compiled = _compile_group_rules(group_rules)
    grouped = []
    used = set()
    for _, patterns in compiled:
        indices = []
        for idx, feat in enumerate(feature_names):
            if idx in used:
                continue
            for pat in patterns:
                if pat.search(feat):
                    indices.append(idx)
                    used.add(idx)
                    break
        if indices:
            grouped.extend(indices)
    for idx in range(len(feature_names)):
        if idx not in used:
            grouped.append(idx)
    return grouped


class SemanticTokenizer(object):
    """
    Semantic tokenizer that maps heterogeneous features into fixed T tokens.
    """

    def __init__(
        self,
        target_tokens,
        d_model,
        embedding_dim,
        semantic_groups=None,
        group_rules=None,
        token_projection="linear",
        name="semantic_tokenizer",
    ):
        self.target_tokens = int(target_tokens)
        self.d_model = int(d_model)
        self.embedding_dim = int(embedding_dim)
        self.semantic_groups = semantic_groups
        self.group_rules = group_rules
        self.token_projection = str(token_projection).lower()
        self.name = str(name)

    def _concat_and_project(self, tensors, scope_name):
        if len(tensors) == 1:
            concat = tensors[0]
        else:
            concat = tf.concat(tensors, axis=-1)
        return tf.compat.v1.layers.dense(concat, units=self.d_model, activation=None, name=scope_name)

    def _pad_or_trim_tokens(self, tokens):
        token_count = tf.shape(tokens)[1]
        if self.target_tokens <= 0:
            return tokens
        if tokens.shape[1] is not None and tokens.shape[1] == self.target_tokens:
            return tokens
        if tokens.shape[1] is not None and tokens.shape[1] > self.target_tokens:
            return tokens[:, : self.target_tokens, :]
        pad_len = self.target_tokens - token_count
        pad = tf.zeros([tf.shape(tokens)[0], pad_len, self.d_model])
        return tf.concat([tokens, pad], axis=1)

    def _build_feature_map(self, dense_embeddings, dense_names, seq_embeddings, seq_names):
        feature_map = {}
        if dense_embeddings is not None and dense_names:
            for idx, name in enumerate(dense_names):
                feature_map[name] = dense_embeddings[:, idx, :]
        if seq_embeddings is not None and seq_names:
            for idx, name in enumerate(seq_names):
                feature_map[name] = seq_embeddings[:, idx, :]
        return feature_map

    def _resolve_group_features(self, group_features, available_names):
        resolved = []
        for raw in group_features:
            if raw in available_names:
                resolved.append(raw)
                continue
            pattern = raw[3:] if raw.startswith("re:") else raw
            if _looks_like_regex(raw):
                regex = re.compile(pattern)
                for name in available_names:
                    if regex.search(name) and name not in resolved:
                        resolved.append(name)
        return resolved

    def tokenize(
        self,
        dense_embeddings,
        dense_feature_names,
        seq_embeddings,
        seq_feature_names,
    ):
        feature_names = []
        if dense_feature_names:
            feature_names.extend(list(dense_feature_names))
        if seq_feature_names:
            feature_names.extend(list(seq_feature_names))
        if not feature_names:
            raise ValueError("SemanticTokenizer needs at least one feature name.")

        feature_map = self._build_feature_map(
            dense_embeddings, dense_feature_names, seq_embeddings, seq_feature_names
        )
        available_names = list(feature_map.keys())

        groups = _normalize_groups(self.semantic_groups)
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if groups:
                tokens = []
                for group_name, group_features in groups:
                    resolved = self._resolve_group_features(group_features, available_names)
                    tensors = [feature_map[name] for name in resolved if name in feature_map]
                    if not tensors:
                        ref = list(feature_map.values())[0]
                        tensors = [tf.zeros([tf.shape(ref)[0], self.embedding_dim])]
                    token = self._concat_and_project(tensors, "token_proj_%s" % _sanitize_group_name(group_name))
                    tokens.append(token)
                stacked = tf.stack(tokens, axis=1)
                stacked = self._pad_or_trim_tokens(stacked)
                stacked.set_shape([None, self.target_tokens, self.d_model])
                return stacked, self.target_tokens

            ordered_names = available_names
            if self.group_rules or DEFAULT_SEMANTIC_GROUP_RULES:
                ordered_indices = _assign_semantic_groups(available_names, self.group_rules)
                ordered_names = [available_names[i] for i in ordered_indices]
            ordered_embeddings = tf.stack([feature_map[name] for name in ordered_names], axis=1)

            feature_count = len(ordered_names)
            target_tokens = self.target_tokens if self.target_tokens > 0 else feature_count
            token_size = int((feature_count + target_tokens - 1) / target_tokens)
            pad_needed = target_tokens * token_size - feature_count
            if pad_needed > 0:
                pad_tensor = tf.zeros([tf.shape(ordered_embeddings)[0], pad_needed, self.embedding_dim])
                ordered_embeddings = tf.concat([ordered_embeddings, pad_tensor], axis=1)
            flat = tf.reshape(
                ordered_embeddings,
                [tf.shape(ordered_embeddings)[0], target_tokens, token_size * self.embedding_dim],
            )
            tokens = tf.compat.v1.layers.dense(flat, units=self.d_model, activation=None, name="token_proj_chunk")
            tokens.set_shape([None, target_tokens, self.d_model])
            return tokens, target_tokens
