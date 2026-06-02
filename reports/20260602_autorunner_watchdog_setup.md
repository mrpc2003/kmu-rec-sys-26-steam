# Autorunner watchdog setup

- cron job id: `272808a2bcca`
- schedule: `*/10 * * * *`
- no_agent: `True`
- runtime script: `/opt/data/scripts/kmu_recsys_autorunner_watchdog.py`
- repo copy: `scripts/kmu_recsys_autorunner_watchdog.py`
- delivery: silent when healthy; prints only on restart/error.
