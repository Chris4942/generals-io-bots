from play.player import Player, DIRECTIONS

class MockMap:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

class MockClient:
    def __init__(self, rows, cols):
        self._map = MockMap(rows, cols)


def run_tests():
    def dumb_general():
        return 0
    
    dumb_client = MockClient(3, 4)
    
    def network_builder():
        return dumb_general
    
    def setup_client():
        return dumb_client
    
    def noop():
        pass
    
    general = dumb_general
    
    player = Player(
        general,
        setup_client,
        "fake_id",
        "fake_name",
        "fake game",
        0,
        network_builder,
        noop,
        "unused",
        None,
        1.0,
        0.9999
    )
    player.client = dumb_client
    
    print("test grid location to vectorized location")
    assert 42 == player.grid_location_to_vectorized_location(2, 1, 3)
    assert 30 == player.grid_location_to_vectorized_location(2, 1, 2)
    assert 18 == player.grid_location_to_vectorized_location(2, 1, 1)
    assert 6 == player.grid_location_to_vectorized_location(2, 1, 0)

    assert (1, 1, 3) == player.vectorized_location_to_grid_location(41)
    assert (1, 1, 2) == player.vectorized_location_to_grid_location(29)
    assert (1, 1, 1) == player.vectorized_location_to_grid_location(17)
    assert (1, 1, 0) == player.vectorized_location_to_grid_location(5)

run_tests()