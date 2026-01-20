import re
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# v2 的更新分组规则，源自新的特征分组方案。
DEFAULT_SEMANTIC_GROUP_RULES = [
    {"name": "customer_dpa", "patterns": [
        r"^commission$",
        r"^commission_rate$",
        r"^dpa_brand_name$",
        r"^dpa_commodity_id$",
        r"^dpa_creative_image$",
        r"^dpa_first_category$",
        r"^dpa_second_category$",
        r"^dpa_shop_id$",
        r"^dpa_shop_title$",
        r"^dpa_title$",
        r"^dpa_type$",
        r"^final_promotion_price$",
        r"^reserve_price$",
        r"^shop_source$",
    ]},
    {"name": "customer_ad_attr", "patterns": [
        r"^idea_content$",
        r"^idea_name$",
        r"^idea_title$",
        r"^image_url$",
        r"^landingurl$",
        r"^logo_url$",
        r"^template_id$",
        r"^template_type$",
        r"^video_url$",
    ]},
    {"name": "customer_adplan_attr", "patterns": [
        r"^ad_idea_id$",
        r"^ad_plan_id$",
        r"^ad_unit_id$",
        r"^combination_un_id$",
        r"^is_new_item$",
        r"^promotion_type$",
        r"^target_type$",
    ]},
    {"name": "customer_task_attr", "patterns": [
        r"^app_first_type$",
        r"^app_level$",
        r"^app_logo$",
        r"^pkg_name$",
        r"^package_name$",
        r"^app_second_type$",
        r"^appdescription$",
        r"^crowd_type$",
        r"^dispatch_center_id$",
        r"^first_industry_id$",
        r"^industry$",
        r"^log10_app_dldcnts$",
        r"^log10_app_size$",
        r"^package_channel_type$",
        r"^product_name$",
        r"^rta_product_code$",
        r"^rta_type$",
        r"^white_crowd_code$",
    ]},
    {"name": "context_area_attr", "patterns": [
        r"^city_level$",
        r"^ip_city$",
        r"^ip_region$",
    ]},
    {"name": "context_bid_attr", "patterns": [
        r"^bid_floor$",
    ]},
    {"name": "context_src_attr", "patterns": [
        r"^app_pkg_src$",
        r"^app_src_first_type$",
        r"^app_src_second_type$",
    ]},
    {"name": "context_traffic_attr", "patterns": [
        r"^adslot_id$",
        r"^adslot_id_type$",
        r"^adx_adslot_id$",
        r"^channel_id$",
        r"^dsp_id$",
        r"^keywords$",
        r"^model_type$",
        r"^pkg_whitelist$",
        r"^source_adslot_type$",
        r"^ssp_adslot_id$",
    ]},
    {"name": "context_device_attr", "patterns": [
        r"^device_brand$",
        r"^device_carrier$",
        r"^device_model$",
        r"^device_os$",
        r"^device_os_version$",
        r"^device_screen_height$",
        r"^device_screen_width$",
        r"^device_size$",
        r"^network$",
    ]},
    {"name": "context_time_attr", "patterns": [
        r"^day_h$",
        r"^is_holiday$",
        r"^week_day$",
    ]},
    {"name": "stat_doc_five", "patterns": [
        r"^doc__key_five__clk_div_imp_cnt_15d$",
        r"^doc__key_five__clk_div_imp_cnt_1d$",
        r"^doc__key_five__clk_div_imp_cnt_30d$",
        r"^doc__key_five__clk_div_imp_cnt_7d$",
        r"^doc__key_five__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_five__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_five__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_five__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_five__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_five__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_five__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_five__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_five__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_five__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_five__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_five__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_five__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_five__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_five__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_five__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_five__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_five__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_five__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_five__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_four", "patterns": [
        r"^doc__key_four__clk_div_imp_cnt_15d$",
        r"^doc__key_four__clk_div_imp_cnt_1d$",
        r"^doc__key_four__clk_div_imp_cnt_30d$",
        r"^doc__key_four__clk_div_imp_cnt_7d$",
        r"^doc__key_four__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_four__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_four__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_four__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_four__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_four__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_four__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_four__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_four__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_four__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_four__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_four__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_four__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_four__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_four__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_four__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_four__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_four__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_four__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_four__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_one", "patterns": [
        r"^doc__key_one__clk_div_imp_cnt_15d$",
        r"^doc__key_one__clk_div_imp_cnt_1d$",
        r"^doc__key_one__clk_div_imp_cnt_30d$",
        r"^doc__key_one__clk_div_imp_cnt_7d$",
        r"^doc__key_one__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_one__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_one__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_one__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_one__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_one__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_one__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_one__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_one__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_one__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_one__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_one__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_one__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_one__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_one__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_one__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_one__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_one__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_one__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_one__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_seven", "patterns": [
        r"^doc__key_seven__clk_div_imp_cnt_15d$",
        r"^doc__key_seven__clk_div_imp_cnt_1d$",
        r"^doc__key_seven__clk_div_imp_cnt_30d$",
        r"^doc__key_seven__clk_div_imp_cnt_7d$",
        r"^doc__key_seven__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_seven__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_seven__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_seven__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_seven__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_seven__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_seven__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_seven__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_seven__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_seven__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_seven__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_seven__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_seven__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_seven__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_seven__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_seven__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_seven__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_seven__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_seven__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_seven__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_six", "patterns": [
        r"^doc__key_six__clk_div_imp_cnt_15d$",
        r"^doc__key_six__clk_div_imp_cnt_1d$",
        r"^doc__key_six__clk_div_imp_cnt_30d$",
        r"^doc__key_six__clk_div_imp_cnt_7d$",
        r"^doc__key_six__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_six__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_six__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_six__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_six__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_six__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_six__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_six__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_six__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_six__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_six__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_six__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_six__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_six__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_six__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_six__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_six__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_six__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_six__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_six__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_three", "patterns": [
        r"^doc__key_three__clk_div_imp_cnt_15d$",
        r"^doc__key_three__clk_div_imp_cnt_1d$",
        r"^doc__key_three__clk_div_imp_cnt_30d$",
        r"^doc__key_three__clk_div_imp_cnt_7d$",
        r"^doc__key_three__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_three__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_three__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_three__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_three__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_three__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_three__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_three__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_three__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_three__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_three__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_three__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_three__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_three__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_three__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_three__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_three__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_three__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_three__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_three__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "stat_doc_two", "patterns": [
        r"^doc__key_two__clk_div_imp_cnt_15d$",
        r"^doc__key_two__clk_div_imp_cnt_1d$",
        r"^doc__key_two__clk_div_imp_cnt_30d$",
        r"^doc__key_two__clk_div_imp_cnt_7d$",
        r"^doc__key_two__kv_city_level_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_city_level_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_city_level_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_day_h_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_day_h_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_day_h_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_brand_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_brand_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_device_brand_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_brand_def_order_div_clk_cnt_30d$",
        r"^doc__key_two__kv_device_carrier_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_carrier_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_model_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_model_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_device_model_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_model_def_order_div_clk_cnt_30d$",
        r"^doc__key_two__kv_device_os_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_os_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_device_os_clk_div_imp_cnt_7d$",
        r"^doc__key_two__kv_device_os_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_os_def_order_div_clk_cnt_1d$",
        r"^doc__key_two__kv_device_os_def_order_div_clk_cnt_30d$",
        r"^doc__key_two__kv_device_os_def_order_div_clk_cnt_7d$",
        r"^doc__key_two__kv_device_os_version_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_os_version_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_device_os_version_clk_div_imp_cnt_7d$",
        r"^doc__key_two__kv_device_os_version_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_device_os_version_def_order_div_clk_cnt_1d$",
        r"^doc__key_two__kv_device_os_version_def_order_div_clk_cnt_7d$",
        r"^doc__key_two__kv_device_size_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_device_size_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_dispatch_center_id_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_ip_city_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_ip_city_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_ip_city_clk_div_imp_cnt_30d$",
        r"^doc__key_two__kv_ip_city_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_ip_city_def_order_div_clk_cnt_7d$",
        r"^doc__key_two__kv_ip_region_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_ip_region_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_ip_region_clk_div_imp_cnt_30d$",
        r"^doc__key_two__kv_ip_region_clk_div_imp_cnt_7d$",
        r"^doc__key_two__kv_ip_region_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_ip_region_def_order_div_clk_cnt_1d$",
        r"^doc__key_two__kv_ip_region_def_order_div_clk_cnt_30d$",
        r"^doc__key_two__kv_ip_region_def_order_div_clk_cnt_7d$",
        r"^doc__key_two__kv_keywords_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_keywords_clk_div_imp_cnt_1d$",
        r"^doc__key_two__kv_keywords_def_order_div_clk_cnt_15d$",
        r"^doc__key_two__kv_network_clk_div_imp_cnt_15d$",
        r"^doc__key_two__kv_network_def_order_div_clk_cnt_15d$",
    ]},
    {"name": "user_stat_dpa", "patterns": [
        r"^user__clk_dpa_commodity_id_dcnt_15d$",
        r"^user__clk_dpa_commodity_id_dcnt_1d$",
        r"^user__clk_dpa_commodity_id_dcnt_30d$",
        r"^user__clk_dpa_commodity_id_dcnt_7d$",
    ]},
    {"name": "user_stat_ad", "patterns": [
        r"^user__clk_combination_un_id_dcnt_1d$",
        r"^user__clk_combination_un_id_dcnt_30d$",
        r"^user__clk_combination_un_id_dcnt_7d$",
        r"^user__clk_div_imp_cnt_7d$",
        r"^user__imp_combination_un_id_dcnt_1d$",
        r"^user__imp_combination_un_id_dcnt_30d$",
        r"^user__imp_combination_un_id_dcnt_7d$",
        r"^user__kv_ad_idea_id_clk_div_imp_cnt_15d$",
        r"^user__kv_ad_idea_id_clk_div_imp_cnt_30d$",
        r"^user__kv_ad_idea_id_clk_div_imp_cnt_7d$",
        r"^user__kv_ad_unit_id_clk_div_imp_cnt_15d$",
        r"^user__kv_ad_unit_id_clk_div_imp_cnt_30d$",
        r"^user__kv_ad_unit_id_clk_div_imp_cnt_7d$",
        r"^user__kv_template_id_clk_div_imp_cnt_15d$",
        r"^user__kv_template_id_clk_div_imp_cnt_30d$",
        r"^user__kv_template_id_clk_div_imp_cnt_7d$",
    ]},
    {"name": "user_stat_traffic", "patterns": [
        r"^user__kv_adslot_id_clk_cnt_1d$",
        r"^user__kv_adslot_id_clk_div_imp_cnt_15d$",
        r"^user__kv_adslot_id_clk_div_imp_cnt_30d$",
        r"^user__kv_adslot_id_clk_div_imp_cnt_7d$",
    ]},
    {"name": "user_stat_perf", "patterns": [
        r"^user__kv_first_category_clk_cnt_15d$",
        r"^user__kv_first_category_clk_cnt_1d$",
        r"^user__kv_first_category_clk_cnt_30d$",
        r"^user__kv_first_category_clk_cnt_7d$",
        r"^user__kv_first_category_clk_div_imp_cnt_15d$",
        r"^user__kv_first_category_clk_div_imp_cnt_1d$",
        r"^user__kv_first_category_clk_div_imp_cnt_30d$",
        r"^user__kv_first_category_clk_div_imp_cnt_7d$",
        r"^user__kv_first_category_imp_cnt_15d$",
        r"^user__kv_first_category_imp_cnt_1d$",
        r"^user__kv_first_category_imp_cnt_30d$",
        r"^user__kv_first_category_imp_cnt_7d$",
    ]},
    {"name": "user_stat_global", "patterns": [
        r"^user__clk_cnt_1d$",
        r"^user__clk_div_imp_cnt_30d$",
        r"^user__kv_day_h_clk_div_imp_cnt_15d$",
        r"^user__kv_day_h_clk_div_imp_cnt_30d$",
        r"^user__kv_day_h_clk_div_imp_cnt_7d$",
    ]},
    {"name": "user_user_attr", "patterns": [
        r"^is_new_user$",
        r"^user_type$",
        r"^age$",
        r"^gender$",
        r"^u_pv$",
    ]},
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
    # 按规则命中顺序排序特征，未命中放末尾。
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
                # 显式 semantic_groups：一个 group 对应一个 token。
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

            # 规则分组：先排序，再均分为 T 个 token。
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
