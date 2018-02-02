import arrow
from google.cloud import bigquery
from google.oauth2.service_account import Credentials


def select_game_streaming_info(game_name, starts_from, ends_to):
    cred = "conf/pym_google_cloud_certificate.json"
    client = bigquery.Client(project="soocii-data", credentials=Credentials.from_service_account_file(cred))
    query = "SELECT * " \
            "FROM `soocii-data.jarvis_prod_backend_media.soocii_streaming_stats` " \
            "WHERE _PARTITIONTIME >= '{}' AND " \
            "_PARTITIONTIME < '{}' AND " \
            "game = '{}'" .format(starts_from, ends_to, game_name)
    timeout = 30
    query_job = client.query(query)
    return [{'owner_soocii_id': i.soocii_id, 'owner_account_id': i.owner,
             'streaming_start_at': arrow.get(i.start_at).timestamp, 'streaming_end_at': arrow.get(i.end_at).timestamp,
             'streaming_url': i.streaming_url, 'streaming_name': i.name}
            for i in query_job.result(timeout=timeout)]

