<?php
header("content-type:text/html;charset=utf-8");
class Server
{
    private $serv;   

    public function __construct() {
        $this->serv = new swoole_server("127.0.0.1", 9501);
        $this->serv->set(array(
            'worker_num' => 1,
            'daemonize' => false,
            'max_request' => 10000,
            'dispatch_mode' => 2,
            'debug_mode'=> 1,
       	    'db_uri' => 'mysql:host=127.0.0.1;dbname=ystcrowds', 
            'db_user' => 'test',
            'db_passwd' => 'test',
            'task_worker_num' => 1
        ));
        $this->serv->on('Start', array($this, 'onStart'));
        $this->serv->on('Connect', array($this, 'onConnect'));
        $this->serv->on('Receive', array($this, 'onReceive'));
        $this->serv->on('Close', array($this, 'onClose'));
        // bind callback
        $this->serv->on('Task', array($this, 'onTask'));
        $this->serv->on('Finish', array($this, 'onFinish'));
        $this->serv->start();
    }
    public function onStart( $serv ) {
        echo "Start\n";
    }
    public function onConnect( $serv, $fd, $from_id ) {
        echo "Client {$fd} connect\n";
    }
    public function onReceive( swoole_server $serv, $fd, $from_id, $data ) {
        echo "Get Message From Client {$fd}:{$data}\n";
        // send a task to task worker.
        $param = array(
            'fd' => $fd,
        'source' => $data
        );
        $serv->task( json_encode( $param ) );
        echo "Continue Handle Worker\n";
    }
    public function onClose( $serv, $fd, $from_id ) {
        echo "Client {$fd} close connection\n";
    }
    public function onTask($serv,$task_id,$from_id, $data) {
        $data_arr=json_decode($data,true);
        $source_data=$data_arr['source'];
        $source = explode(' ', $source_data);
        
        $trainid = $source[1];
        $command = './tools/preprocess.py --id '.$trainid;
        exec($command);
       
        return "Task {$task_id}'s result";
    }

    
    public function onFinish($serv,$task_id, $data) {


    }

    
    /**
    * send message by swoole
    * @param string $content the command
    * return boolean true if shut down the swoole_client successfully
    */
    private function sendtagbyswoole($content){
        $client = new swoole_client(SWOOLE_SOCK_TCP, SWOOLE_SOCK_SYNC);
        $client->connect('127.0.0.1', 9501, 0.5, 0);
        $client->send($content);
        return $client -> close();
    } 

}
$server = new Server();
$server->start();
