from __future__ import unicode_literals
from .dag import get_outgoing_edges, topo_sort
from ._utils import basestring
from builtins import str
from functools import reduce
import collections
import copy
import operator
import subprocess

from ._ffmpeg import (
    input,
    output,
)
from .nodes import (
    get_stream_spec_nodes,
    FilterNode,
    GlobalNode,
    InputNode,
    OutputNode,
    output_operator,
)


class Error(Exception):
    def __init__(self, cmd, stdout, stderr):
        super(Error, self).__init__('{} error (see stderr output for detail)'.format(cmd))
        self.stdout = stdout
        self.stderr = stderr


def _convert_kwargs_to_cmd_line_args(kwargs):
    args = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        args.append('-{}'.format(k))
        if v is not None:
            args.append('{}'.format(v))
    return args


def _get_input_args(input_node):
    if input_node.name == input.__name__:
        kwargs = copy.copy(input_node.kwargs)
        filename = kwargs.pop('filename')
        fmt = kwargs.pop('format', None)
        video_size = kwargs.pop('video_size', None)
        args = []
        if fmt:
            args += ['-f', fmt]
        if video_size:
            args += ['-video_size', '{}x{}'.format(video_size[0], video_size[1])]
        args += _convert_kwargs_to_cmd_line_args(kwargs)
        args += ['-i', filename]
    else:
        raise ValueError('Unsupported input node: {}'.format(input_node))
    return args


def _format_input_stream_name(stream_name_map, edge, is_final_arg=False):
    prefix = stream_name_map[edge.upstream_node, edge.upstream_label]
    if not edge.upstream_selector:
        suffix = ''
    else:
        suffix = ':{}'.format(edge.upstream_selector)
    if is_final_arg and isinstance(edge.upstream_node, InputNode):
        ## Special case: `-map` args should not have brackets for input
        ## nodes.
        fmt = '{}{}'
    else:
        fmt = '[{}{}]'
    return fmt.format(prefix, suffix)


def _format_output_stream_name(stream_name_map, edge):
    return '[{}]'.format(stream_name_map[edge.upstream_node, edge.upstream_label])


def _get_filter_spec(node, outgoing_edge_map, stream_name_map):
    incoming_edges = node.incoming_edges
    outgoing_edges = get_outgoing_edges(node, outgoing_edge_map)
    inputs = [_format_input_stream_name(stream_name_map, edge) for edge in incoming_edges]
    outputs = [_format_output_stream_name(stream_name_map, edge) for edge in outgoing_edges]
    filter_spec = '{}{}{}'.format(''.join(inputs), node._get_filter(outgoing_edges), ''.join(outputs))
    return filter_spec


def _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map):
    stream_count = 0
    for upstream_node in filter_nodes:
        outgoing_edge_map = outgoing_edge_maps[upstream_node]
        for upstream_label, downstreams in list(outgoing_edge_map.items()):
            if len(downstreams) > 1:
                # TODO: automatically insert `splits` ahead of time via graph transformation.
                raise ValueError(
                    'Encountered {} with multiple outgoing edges with same upstream label {!r}; a '
                    '`split` filter is probably required'.format(upstream_node, upstream_label))
            stream_name_map[upstream_node, upstream_label] = 's{}'.format(stream_count)
            stream_count += 1


def _get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map):
    _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map)
    filter_specs = [_get_filter_spec(node, outgoing_edge_maps[node], stream_name_map) for node in filter_nodes]
    return ';'.join(filter_specs)


def _get_global_args(node):
    return list(node.args)


def _get_output_args(node, stream_name_map):
    if node.name != output.__name__:
        raise ValueError('Unsupported output node: {}'.format(node))
    args = []

    if len(node.incoming_edges) == 0:
        raise ValueError('Output node {} has no mapped streams'.format(node))

    for edge in node.incoming_edges:
        # edge = node.incoming_edges[0]
        stream_name = _format_input_stream_name(stream_name_map, edge, is_final_arg=True)
        if stream_name != '0' or len(node.incoming_edges) > 1:
            args += ['-map', stream_name]

    kwargs = copy.copy(node.kwargs)
    filename = kwargs.pop('filename')
    if 'format' in kwargs:
        args += ['-f', kwargs.pop('format')]
    if 'video_bitrate' in kwargs:
        args += ['-b:v', str(kwargs.pop('video_bitrate'))]
    if 'audio_disable' in kwargs:
        if kwargs.pop('audio_disable') is True:
            args += ['-an']
    if 'audio_bitrate' in kwargs:
        args += ['-b:a', str(kwargs.pop('audio_bitrate'))]
    if 'video_size' in kwargs:
        video_size = kwargs.pop('video_size')
        if not isinstance(video_size, basestring) and isinstance(video_size, collections.Iterable):
            video_size = '{}x{}'.format(video_size[0], video_size[1])
        args += ['-video_size', video_size]
    args += _convert_kwargs_to_cmd_line_args(kwargs)
    args += [filename]
    return args

def _get_frame_size(stream_spec):
    nodes = get_stream_spec_nodes(stream_spec)
    sorted_nodes, outgoing_edge_maps = topo_sort(nodes)

    # calculate from the output stream
    output_node = next(node for node in sorted_nodes if isinstance(node, OutputNode))
    kwargs = copy.copy(output_node.kwargs)

    width = 0
    height = 0

    if 'video_size' in kwargs:
        width, height = kwargs.pop('video_size')
        pix_fmt = kwargs.pop('pix_fmt')

    else:
        # probe the input stream instead
        from ._probe import probe
        input_node = next(node for node in sorted_nodes if isinstance(node, InputNode))
        kwargs = copy.copy(input_node.kwargs)

        filename = str(input_node.kwargs.pop('filename'))

        probe = probe(filename)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        pix_fmt = str(video_info['pix_fmt'])

    if pix_fmt == 'yuvj420p':
        depth = 3
    elif pix_fmt == 'rgb24':
        depth = 3
    elif pix_fmt == 'gray':
        depth = 1
    else:
        depth = 3

    return width * height * depth

@output_operator()
def get_args(stream_spec, overwrite_output=False):
    """Build command-line arguments to be passed to ffmpeg."""
    nodes = get_stream_spec_nodes(stream_spec)
    args = []
    # TODO: group nodes together, e.g. `-i somefile -r somerate`.
    sorted_nodes, outgoing_edge_maps = topo_sort(nodes)
    input_nodes = [node for node in sorted_nodes if isinstance(node, InputNode)]
    output_nodes = [node for node in sorted_nodes if isinstance(node, OutputNode)]
    global_nodes = [node for node in sorted_nodes if isinstance(node, GlobalNode)]
    filter_nodes = [node for node in sorted_nodes if isinstance(node, FilterNode)]
    stream_name_map = {(node, None): str(i) for i, node in enumerate(input_nodes)}
    filter_arg = _get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map)
    args += reduce(operator.add, [_get_input_args(node) for node in input_nodes])
    if filter_arg:
        args += ['-filter_complex', filter_arg]
    args += reduce(operator.add, [_get_output_args(node, stream_name_map) for node in output_nodes])
    args += reduce(operator.add, [_get_global_args(node) for node in global_nodes], [])
    if overwrite_output:
        args += ['-y']
    return args


@output_operator()
def compile(stream_spec, cmd='ffmpeg', overwrite_output=False):
    """Build command-line for invoking ffmpeg.

    The :meth:`run` function uses this to build the commnad line
    arguments and should work in most cases, but calling this function
    directly is useful for debugging or if you need to invoke ffmpeg
    manually for whatever reason.

    This is the same as calling :meth:`get_args` except that it also
    includes the ``ffmpeg`` command as the first argument.
    """
    if isinstance(cmd, basestring):
        cmd = [cmd]
    elif type(cmd) != list:
        cmd = list(cmd)
    return cmd + get_args(stream_spec, overwrite_output=overwrite_output)


@output_operator()
def run(
        stream_spec, cmd='ffmpeg', capture_stdout=False, capture_stderr=False, input=None,
        quiet=False, overwrite_output=False):
    """Ivoke ffmpeg for the supplied node graph.

    Args:
        capture_stdout: if True, capture stdout (to be used with
            ``pipe:`` ffmpeg outputs).
        capture_stderr: if True, capture stderr.
        quiet: shorthand for setting ``capture_stdout`` and ``capture_stderr``.
        input: text to be sent to stdin (to be used with ``pipe:``
            ffmpeg inputs)
        **kwargs: keyword-arguments passed to ``get_args()`` (e.g.
            ``overwrite_output=True``).

    Returns: (out, err) tuple containing captured stdout and stderr data.
    """
    args = compile(stream_spec, cmd, overwrite_output=overwrite_output)
    stdin_stream = subprocess.PIPE if input else None
    stdout_stream = subprocess.PIPE if capture_stdout or quiet else None
    stderr_stream = subprocess.PIPE if capture_stderr or quiet else None
    p = subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream)
    out, err = p.communicate(input)
    retcode = p.poll()
    if retcode:
        raise Error('ffmpeg', out, err)
    return out, err


@output_operator(name='stream')
def stream(stream_spec, cmd='ffmpeg', capture_stderr=False, input=None, quiet=False, overwrite_output=False):
    """ Invoke ffmpeg for the supplied node graph, streaming frame by frame.

    """
    args = compile(stream_spec, cmd, overwrite_output=overwrite_output)

    # calculate framezie
    framesize = _get_frame_size(stream_spec, args)

    stdin_stream = subprocess.PIPE if input else None
    stdout_stream = subprocess.PIPE
    stderr_stream = subprocess.PIPE if capture_stderr or quiet else None
    p = subprocess.Popen(args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream)

    while p.poll() is None:
        yield _read_frame(p, framesize)


def _read_frame(process, framesize):
    return process.stdout.read(framesize)


__all__ = [
    'compile',
    'Error',
    'get_args',
    'run',
    'stream'
]
