import collections

StaffLine = collections.namedtuple('StaffLine', 'which points_along')
Accidental = collections.namedtuple('Accidental', 'type center_position')
Note = collections.namedtuple('Note', 'center_position bounding_box pitch duration')