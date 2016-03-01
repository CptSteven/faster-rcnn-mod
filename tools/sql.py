#!/usr/bin/env python
#-*- coding: utf-8 -*-

import MySQLdb
import _mysql_exceptions as DB_EXCEPTION

class ConnException(Exception):
    def __init__(self,text):
        Exception.__init__()
        self.msg = text
   
    def __str__(self):
        return self.msg


class MySQLConnection(object):
    def __init__(self,host=None, user='', passwd='', dbname=None):
        self._host = host
        self._user = user
        self._passwd = passwd
        self._dbname = dbname
        self._cxn = None
        self._cursor = None

    def connect(self):
        try:
            self._cxn = MySQLdb.connect(self._host, self._user, \
                                       self._passwd, self._dbname, \
                                       charset='utf8', port=3306 )
            if not self._cxn:
                raise ConnException('Cannot connect to MySQL server.')
            self._cursor = self._cxn.cursor()
        except ConnException as e:
            print e
            return False
        except Exception as e:
            print 'Unknown error.'
            print e
            return False
        return True

    def getCursor(self):
        return self._cursor
 
    def commit(self):
        return self._cxn.commit()

    def rollback(self):
        return NotImplemented

    def close(self):
        if self._cursor != None:
            self._cursor.close()
        if self._cxn != None:
            self._cxn.close()

    def query(self, sql, args=None, many=False):
        affected_rows = 0
        if not many:
            if args == None:
                affected_rows = self._cursor.execute(sql)
            else:
                affected_rows = self._cursor.execute(sql, args)
        else:
            if args == None:
                affected_rows = self._cursor.executemany(sql)
            else:
                affected_rows = self._cursor.executemany(sql, args)
        return affected_rows

    def fetchAll(self):
        return self._cursor.fetchall()

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, host):
        self._host = host

    @property
    def user(self):
        return self._host

    @user.setter
    def user(self, user):
        self._user = user 

    @property
    def password(self):
        return self._passwd

    @password.setter
    def password(self, passwd):
        self._passwd = passwd
    
    @property
    def dbname(self):
        return self._dbname

    @dbname.setter
    def dbname(self, dbname):
        self._dbname = dbname 
    
    
if __name__ == '__main__':
    conn = MySQLConnection()
    conn.host = '192.168.1.90'
    conn.user = 'test'
    conn.password = 'test'
    conn.dbname = 'zb_label'

    status = {}
    status['untrained'] = 1
    status['training'] = 2
    status['trained'] = 3
    conn.connect()
    sqlstr = {}
    sqlstr['getinst1'] = 'select * from zb_train where id = {} \
                          order by createtime asc'.format(11)#status['untrained'])
    #sqlstr['getinst1'] = 'select * from zb_train where status = {} \
    #                      and machine_id = {}'.format(status['untrained'],mid)
    conn.query(sqlstr['getinst1'])    
    data = conn.fetchAll()
    conn.close()
#    for i in range(len(data[0])):
#        print data[0][i]
    print data[0]

