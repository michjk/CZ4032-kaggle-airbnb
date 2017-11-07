#train test 2 attribute
country_destination_name = 'country_destination'
country_name = 'country'
id_name = 'id'
date_first_booking_name = "date_first_booking"
date_account_created_name = "date_account_created"
timestamp_first_active_name = "timestamp_first_active"
gender_name = "gender"
signup_method_name = 'signup_method'
signup_flow_name = 'signup_flow'
language_name = 'language'
affiliate_channel_name = 'affiliate_channel'
affiliate_provider_name = 'affiliate_provider'
first_affiliate_tracked_name = 'first_affiliate_tracked'
signup_app_name = 'signup_app'
first_device_type_name = 'first_device_type'
first_browser_name = 'first_browser'
age_name = 'age'
action_name = 'action'
action_type_name = 'action_type'
action_detail_name = 'action_detail'
device_type_name = 'device_type'
secs_elapsed_name = 'secs_elapsed'

#nominal attribute
nominal_train_column_list = [gender_name, signup_method_name, signup_flow_name, language_name, affiliate_channel_name, affiliate_provider_name, first_affiliate_tracked_name, signup_app_name, first_device_type_name, first_browser_name]
nominal_session_column_list = [action_name, action_type_name, action_detail_name, device_type_name, secs_elapsed_name]
nominal_train_session_column_list = nominal_train_column_list + nominal_session_column_list
