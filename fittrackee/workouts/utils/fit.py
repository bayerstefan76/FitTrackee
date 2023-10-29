from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

from garmin_fit_sdk import Decoder, Stream, Profile
from gpxpy.gpx import GPXBounds, GPXTrackPoint
import gpxpy

from ..exceptions import InvalidFitException, WorkoutFitException
from .weather import WeatherService

weather_service = WeatherService()

SEMICIRCLES_TO_GPS = (1 / ((2 ** 32) / 360))
MESG_NUM_WITH_LONGITUDE_LATTITUDE = [18, 19, 20]


def get_field_names_with_unit(mesg_profile, unit):
    field_list = [v['name'] for (k, v) in mesg_profile['fields'].items() if v['units'] == unit]
    return field_list


def mesg_listener(mesg_num, message):
    if mesg_num in MESG_NUM_WITH_LONGITUDE_LATTITUDE:
        # longitude/lattidude semircircles to degrees
        field_list = get_field_names_with_unit(Profile['messages'][mesg_num], 'semicircles')
        for key in message:
            if key in field_list:
                message[key] *= SEMICIRCLES_TO_GPS


class MinimumMaximum(NamedTuple):
    minimum: Optional[float]
    maximum: Optional[float]


class UphillDownhill(NamedTuple):
    uphill: float
    downhill: float


class MovingData(NamedTuple):
    elapsed_time: float  # seconds
    timer_duration: float  # seconds
    moving_duration: float  # seconds
    stop_duration: float  # seconds
    total_distance: float  # meters
    max_speed: float  # km/h
    avg_speed: float  # km/h


class FitFile:
    def __init__(self, filename):
        self.filename = filename
        # read file
        stream = Stream.from_file(filename)
        decoder = Decoder(stream)
        self.messages, errors = decoder.read(mesg_listener=mesg_listener)

    def is_empty(self):
        if len(self.messages['record_mesgs']) == 0:
            return True
        else:
            return False

    def get_max_speed(self, session_num: Union[int, None] = None) -> Union[float, None]:
        max_speed = 0
        enhanced = False
        not_there = True
        num_sessions = self.messages['activity_mesgs'][0]['num_sessions']
        if session_num is None:
            session_range = range(num_sessions)
        else:
            session_range = [session_num]
        for session_idx in session_range:
            session = self.messages['session_mesgs'][session_idx]
            if 'enhanced_max_speed' in session.keys():
                enhanced = True
                not_there = False
                if session['enhanced_max_speed'] > max_speed:
                    max_speed = session['enhanced_max_speed']
            elif 'max_speed' in session.keys():
                not_there = False
                if session['max_speed'] > max_speed:
                    max_speed = session['max_speed']
        if not_there:
            return None

        if enhanced:
            max_speed *= 3.6

        return max_speed

    def get_start_time(self, session_idx: int = 0) -> datetime:
        return self.messages['session_mesgs'][session_idx]['start_time']

    def get_elevation_extremes(self, session_num: Union[int, None] = None) -> MinimumMaximum:
        # browse through records
        num_sessions = self.messages['activity_mesgs'][0]['num_sessions']
        if 'record_mesgs' not in self.messages.keys():
            return MinimumMaximum(None, None)
        elif len(self.messages['record_mesgs']) == 0:
            return MinimumMaximum(None, None)
        else:
            if session_num is None:
                session_range = range(num_sessions)
            else:
                session_range = [session_num]
            for session_idx in session_range:
                records = self.session_get_position_records(session_idx)
                minimum = records[0]['enhanced_altitude']
                maximum = minimum
                for i in range(1, len(records), 1):
                    if minimum > records[i]['enhanced_altitude']:
                        minimum = records[i]['enhanced_altitude']
                    if maximum < records[i]['enhanced_altitude']:
                        maximum = records[i]['enhanced_altitude']
            return MinimumMaximum(minimum, maximum)

    def get_uphill_downhill(self, session_num: Union[int, None] = None) -> MinimumMaximum:
        num_sessions = self.messages['activity_mesgs'][0]['num_sessions']
        if num_sessions == 0:
            return UphillDownhill(None, None)
        uphill = 0
        downhill = 0
        if session_num is None:
            session_range = range(num_sessions)
        else:
            session_range = [session_num]
        for i in session_range:
            session = self.messages['session_mesgs'][i]
            if 'total_ascent' in session.keys():
                uphill += session['total_ascent']
            if 'total_descent' in session.keys():
                downhill += session['total_descent']
        return UphillDownhill(uphill, downhill)

    def get_moving_data(self, session_num: Union[int, None] = None) -> Optional[MovingData]:
        num_sessions = self.messages['activity_mesgs'][0]['num_sessions']

        if num_sessions == 0:
            return MovingData(0, 0, 0, 0, 0, 0, 0)
        else:
            elapsed_time = 0
            timer_duration = 0
            total_distance = 0
            if session_num is None:
                session_range = range(num_sessions)
            else:
                session_range = [session_num]
            for i in session_range:
                session = self.messages['session_mesgs'][i]
                total_distance += session['total_distance']
                elapsed_time += session['total_elapsed_time']
                timer_duration += session['total_timer_time']
            avg_speed = (total_distance / timer_duration) * 3.6
            moving_duration = timer_duration
            stop_duration = elapsed_time - timer_duration
            max_speed = self.get_max_speed(session_num)
            return MovingData(elapsed_time, timer_duration, moving_duration, stop_duration, total_distance, max_speed,
                              avg_speed)

    def session_get_position_records(self, session_num: Union[int, None] = None) -> List:
        if session_num is not None:
            start_time = self.messages['session_mesgs'][session_num]['start_time']
            end_time = self.messages['session_mesgs'][session_num]['timestamp']
            records = [message for message in self.messages['record_mesgs'] if
                       message['timestamp'] >= start_time and message[
                           'timestamp'] < end_time and 'position_lat' in message.keys() and 'position_long' in message.keys()]
        else:
            records = [message for message in self.messages['record_mesgs'] if
                       'position_lat' in message.keys() and 'position_long' in message.keys()]

        return records

    def get_session_bounds(self, session_idx: int) -> Optional[GPXBounds]:
        result: Optional[GPXBounds] = None
        session = self.messages['session_mesgs'][session_idx]
        if 'swc_lat' in session.keys():
            result = GPXBounds(session['swc_lat'], session['nec_lat'], session['swc_long'], session['nec_long'])
        return result

    def get_bounds(self, session_num: Union[int, None] = None) -> Optional[GPXBounds]:
        num_sessions = self.messages['activity_mesgs'][0]['num_sessions']
        result: Optional[GPXBounds] = None
        if num_sessions > 0:
            if session_num is None:
                session_range = range(num_sessions)
            else:
                session_range = [session_num]
            for session_idx in session_range:
                session_bounds = self.get_session_bounds(session_idx)
                if result is None:
                    result = session_bounds
                else:
                    result = session_bounds.max_bounds(result)

        return result

    def session_to_gpx(self, session_idx: Union[int, None] = None):
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        records = self.session_get_position_records(session_idx)
        for record in records:
            gpx_segment.points.append(
                gpxpy.gpx.GPXTrackPoint(
                    record['position_lat'], record['position_long'], elevation=record['enhanced_altitude']
                )
            )

        return gpx


def open_fit_file(fit_file: str) -> Optional[FitFile]:
    fit_file = FitFile(fit_file)
    if fit_file.is_empty():
        return None
    return fit_file


def get_fit_data(
        fit_file: FitFile
) -> Dict:
    """
    Returns data from parsed fit file
    """
    fit_data: Dict[str, Any] = {
        'max_speed': fit_file.get_max_speed(),
        'start': fit_file.get_start_time(),
    }

    ele = fit_file.get_elevation_extremes()
    fit_data['elevation_max'] = ele.maximum
    fit_data['elevation_min'] = ele.minimum

    # fit file contains elevation data (<ele> element)
    if ele.maximum is not None:
        hill = fit_file.get_uphill_downhill()
        fit_data['uphill'] = hill.uphill
        fit_data['downhill'] = hill.downhill
    else:
        fit_data['uphill'] = None
        fit_data['downhill'] = None

    moving_data = fit_file.get_moving_data()
    if moving_data:
        fit_data['moving_time'] = timedelta(seconds=moving_data.moving_duration)
        fit_data['stop_time'] = timedelta(seconds=moving_data.stop_duration)
        fit_data['distance'] = moving_data.total_distance / 1000
        fit_data['average_speed'] = moving_data.avg_speed
        fit_data['duration'] = timedelta(seconds=moving_data.elapsed_time)

    return fit_data


def get_fit_data_session(
        fit_file: FitFile,
        session_idx: int
) -> Dict:
    """
    Returns data from parsed fit file
    """
    fit_data: Dict[str, Any] = {
        'max_speed': fit_file.get_max_speed(session_idx),
        'start': fit_file.get_start_time(session_idx),
    }

    ele = fit_file.get_elevation_extremes(session_idx)
    fit_data['elevation_max'] = ele.maximum
    fit_data['elevation_min'] = ele.minimum

    # fit file contains elevation data (<ele> element)
    if ele.maximum is not None:
        hill = fit_file.get_uphill_downhill(session_idx)
        fit_data['uphill'] = hill.uphill
        fit_data['downhill'] = hill.downhill
    else:
        fit_data['uphill'] = None
        fit_data['downhill'] = None

    moving_data = fit_file.get_moving_data(session_idx)
    if moving_data:
        fit_data['moving_time'] = timedelta(seconds=moving_data.moving_duration)
        fit_data['stop_time'] = timedelta(seconds=moving_data.stop_duration)
        fit_data['distance'] = moving_data.total_distance / 1000  # km
        fit_data['average_speed'] = moving_data.avg_speed
        fit_data['duration'] = timedelta(seconds=moving_data.elapsed_time)

    return fit_data


def get_fit_info(
        fit_file: str,
        update_map_data: Optional[bool] = True,
        update_weather_data: Optional[bool] = True,
) -> Tuple:
    """
    Parse and return fit, map and weather data from fit file
    """
    try:
        fit = open_fit_file(fit_file)
    except Exception:
        raise InvalidFitException('error', 'fit file is invalid')
    if fit is None:
        raise InvalidFitException('error', 'no tracks in fit file')

    #fit_data: Dict = {'name': fit.messages['activity_mesgs'][0]['event'], 'segments': []}
    # ToDo: use correct sports identifier from fit file
    fit_data: Dict = {'name': None, 'segments': []}
    max_speed = 0.0
    start: Optional[datetime] = None
    map_data = []
    weather_data = []
    num_sessions = fit.messages['activity_mesgs'][0]['num_sessions']
    segments_nb = num_sessions
    prev_seg_last_point = None
    no_stopped_time = timedelta(seconds=0)
    stopped_time_between_seg = no_stopped_time

    for session_idx in range(segments_nb):
        segment_start: Optional[datetime] = fit.messages['session_mesgs'][session_idx]['start_time']
        segment_records = fit.session_get_position_records(session_idx)
        segment_points_nb = len(segment_records)
        for point_idx, point in enumerate(segment_records):
            if 'timestamp' not in point.keys():
                raise InvalidFitException(
                    'error', '<timestamp> is missing in fit file'
                )
            if point_idx == 0:
                segment_start = point['timestamp']
                # first fit point => get weather
                if start is None:
                    start = point['timestamp']
                    if point['timestamp'] and update_weather_data:
                        weather_data.append(
                            weather_service.get_weather(GPXTrackPoint(point['position_lat'], point['position_long'],
                                                                      point['timestamp'])))

                # if a previous segment exists, calculate stopped time between
                # the two segments
                if prev_seg_last_point:
                    stopped_time_between_seg += (
                            point.time - prev_seg_last_point
                    )

            # last segment point
            if point_idx == (segment_points_nb - 1):
                prev_seg_last_point = point['timestamp']

                # last fit point => get weather
                if session_idx == (segments_nb - 1) and update_weather_data:
                    weather_data.append(
                        weather_service.get_weather(GPXTrackPoint(point['position_lat'], point['position_long'],
                                                                  point['timestamp'])))

            if update_map_data:
                map_data.append([point['position_long'], point['position_lat']])
        moving_data = fit.get_moving_data(session_idx)
        if moving_data:
            calculated_max_speed = moving_data.max_speed
            segment_max_speed = (
                calculated_max_speed if calculated_max_speed else 0
            )

            if segment_max_speed > max_speed:
                max_speed = segment_max_speed
        else:
            segment_max_speed = 0.0

        # ToDo: this is global only atm, adapt to activities with more than one segment
        segment_data = get_fit_data_session(fit, session_idx)
        segment_data['idx'] = session_idx
        fit_data['segments'].append(segment_data)

    full_fit_data = get_fit_data(
        fit
    )
    fit_data = {**fit_data, **full_fit_data}

    if update_map_data:
        bounds = fit.get_bounds()
        fit_data['bounds'] = (
            [
                bounds.min_latitude,
                bounds.min_longitude,
                bounds.max_latitude,
                bounds.max_longitude,
            ]
            if bounds
            else []
        )

    return fit_data, map_data, weather_data


def fit_time_difference(dt1: datetime, dt2: Optional[datetime]) -> float:
    if dt2 is None:
        return 0
    else:
        delta = dt1 - dt2
        return delta.total_seconds()


def get_fit_chart_data(
        fit_file: str, session_id: Optional[int] = None
) -> Optional[List]:
    """
    Return data needed to generate chart with speed and elevation
    """
    fit = open_fit_file(fit_file)
    if fit is None:
        return None

    chart_data = []

    num_sessions = fit.messages['activity_mesgs'][0]['num_sessions']
    first_point = None

    if num_sessions > 0:
        if session_id is None:
            session_range = range(num_sessions)
        else:
            session_range = [session_id]
    for session_idx in session_range:
        records = fit.session_get_position_records(session_idx)
        for record in records:
            if first_point is None:
                first_point = record
            data = {
                'distance': (
                    record['distance']
                ),
                'duration': fit_time_difference(record['timestamp'], first_point['timestamp']),
                'latitude': record['position_lat'],
                'longitude': record['position_long'],
                'speed': record['enhanced_speed'] * 3.6,
                'time': record['timestamp'],
            }
            if record['enhanced_altitude']:
                data['elevation'] = record['enhanced_altitude']
            chart_data.append(data)

    return chart_data


def extract_segment_from_fit_file(
        fit_file_name: str, session_id: Union[int, None] = None
) -> Optional[str]:
    """
    Returns session location data in gpx xml format from a fit file
    """
    fit = FitFile(fit_file_name)
    num_sessions = fit.messages['activity_mesgs'][0]['num_sessions']
    if num_sessions == 0:
        return None

    gpx = fit.session_to_gpx(session_id)

    return gpx.to_xml()
