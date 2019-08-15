//import com.webank.ai.fate.board.ssh.SFTPUtils;
//import com.webank.ai.fate.board.ssh.SshService;
//import org.junit.Test;
//import org.junit.runner.RunWith;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.boot.test.context.SpringBootTest;
//import org.springframework.test.context.junit4.SpringRunner;
//
///**
// * @Description TODO
// * @Author kaideng
// **/
//@SpringBootTest(classes=com.webank.ai.fate.board.bootstrap.Bootstrap.class)
//// 让 JUnit 运行 Spring 的测试环境， 获得 Spring 环境的上下文的支持
//@RunWith(SpringRunner.class)
//
//public class TestFileTransfer {
//    @Autowired
//    SshService sshService;
//
//    @Test
//    public  void  testFileTransfer(){
//
//       SshService.SSHInfo  sshInfo = sshService.getSSHInfo("localhost");
//
//        SFTPUtils.batchDownLoadFile(sshInfo,"/Users/kaideng/test","/Users/kaideng/test1/",null,null,false);
//
//
//    }
//
//
//}
