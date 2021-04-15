"""
Bridges Python 2 and 3
"""
# https://stackoverflow.com/questions/27863832/calling-python-2-script-from-python-3
import execnet

def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()