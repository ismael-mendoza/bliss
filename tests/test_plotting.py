def test_plotting(sleep_setup):
    # just to test `make_validation_plots` works.
    overrides = dict(
        model="sleep_star_basic", dataset="test_plotting", training="test_plotting"
    )
    sleep_setup.get_trained_sleep(overrides)
