import time

import requests


def get_with_retry(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    max_attempts: int = 4,
    backoff_seconds: float = 1.0,
) -> requests.Response:
    """Send a GET request with retries and exponential backoff."""

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
            )

            response.raise_for_status()

            return response

        except requests.RequestException as error:
            last_error = error

            if attempt == max_attempts:
                break

            wait_seconds = backoff_seconds * (2 ** (attempt - 1))

            print(
                f"Request failed on attempt {attempt}/{max_attempts}. "
                f"Retrying in {wait_seconds:.0f} seconds..."
            )

            time.sleep(wait_seconds)

    raise requests.RequestException(
        f"Request failed after {max_attempts} attempts: {url}"
    ) from last_error