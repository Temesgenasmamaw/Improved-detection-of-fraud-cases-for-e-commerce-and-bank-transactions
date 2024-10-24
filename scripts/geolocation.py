import pandas as pd
import socket
import struct
class GeolocationDataPreprocessing:

    def __init__(self):
        pass
    def ip_to_int(self,ip):
        
        
        """Convert an IP address to its integer representation.
        socket.inet_aton(ip)->Converts the IP address string into its binary format (a 4-byte representation) from the socket library.
                            For example, "192.168.0.1" becomes b'\xc0\xa8\x00\x01', which is the binary equivalent of the IP address
        struct.unpack("!I", ...)->This function unpacks the 4-byte binary data into an unsigned 32-bit integer where "!": Network byte order.add() 
                                For example, b'\xc0\xa8\x00\x01' gets unpacked to 3232235521 (which is the integer equivalent of "192.168.0.1").
                                
        """
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except socket.error:
            return None